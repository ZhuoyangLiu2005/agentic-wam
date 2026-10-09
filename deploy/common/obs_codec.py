#!/usr/bin/env python
"""Image encode/decode for the observation payload. Needs numpy + Pillow (both sides).

The decode path is deliberately byte-identical to what `deploy/deploy_server.py` already does:

    np.asarray(Image.open(BytesIO(b64decode(x))).convert("RGB"), dtype=np.uint8)

so switching to this module changes nothing the model sees. The encode side is where the remote
line earns its keep.

PAYLOAD SIZE — the reason this module exists
--------------------------------------------
Today's client sends 3 base64 PNGs at native resolution: roughly 2 MB/request, ~16 Mbps sustained
at 1 replan/s. On loopback that is free; over a real link the transfer alone can dominate the
control budget. JPEG q90 cuts it ~10-30x.

    format   640x480 x3, base64      typical /act payload
    png            ~2.0 MB           what deploy_client_example.py sends today
    jpeg q90       ~0.18 MB          recommended for the remote line

JPEG is lossy, which is a real (if small) domain shift: the model trained on frames decoded from
H.264 video, so it has seen block artifacts of a similar character. Validate with the replay
harness before trusting it on hardware -- `--format png` stays available as the reference path.

RESIZING — leave it on the server
---------------------------------
The server applies the training-identical `ResizeSmallestSideAspectPreserving -> CenterCrop(320x224)`.
`max_smallest_side` here can pre-shrink client-side to cut bandwidth further, but any mismatch with
the server transform breaks pixel-parity with training silently. Default is None (send native, let
the server do it) -- one source of truth. Only turn it on after replay validation.
"""
import base64
import io
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from .contract import CAMERAS, ContractError, IMAGE_FORMATS


def encode_image(rgb: np.ndarray, fmt: str = "jpeg", quality: int = 90,
                 max_smallest_side: Optional[int] = None) -> Dict[str, str]:
    """HxWx3 uint8 RGB -> {"format":..., "data": base64}. See module docstring on max_smallest_side."""
    if fmt not in IMAGE_FORMATS:
        raise ContractError(f"unsupported image format {fmt!r}, expected one of {list(IMAGE_FORMATS)}")
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ContractError(f"expected HxWx3 RGB, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    if max_smallest_side is not None:
        w, h = img.size
        s = min(w, h)
        if s > max_smallest_side:
            scale = max_smallest_side / float(s)
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)

    buf = io.BytesIO()
    if fmt == "jpeg":
        img.save(buf, format="JPEG", quality=int(quality), subsampling=0)  # 4:4:4, no chroma loss
    else:
        img.save(buf, format="PNG", compress_level=1)                     # speed over ratio
    return {"format": fmt, "data": base64.b64encode(buf.getvalue()).decode()}


def decode_image(entry: Dict[str, str]) -> np.ndarray:
    """{"format":..., "data": base64} -> HxWx3 uint8 RGB. Matches deploy_server.py::_decode."""
    try:
        raw = base64.b64decode(entry["data"])
    except Exception as e:
        raise ContractError(f"image data is not valid base64: {e}")
    try:
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    except Exception as e:
        raise ContractError(f"image bytes could not be decoded as {entry.get('format')}: {e}")


def encode_observation(images: Dict[str, np.ndarray], fmt: str = "jpeg", quality: int = 90,
                       max_smallest_side: Optional[int] = None) -> Dict[str, Dict[str, str]]:
    """{camera: HxWx3 uint8} -> the `images` field of an ActRequest. Requires all of CAMERAS."""
    missing = [c for c in CAMERAS if c not in images]
    if missing:
        raise ContractError(f"missing cameras {missing} (need all of {list(CAMERAS)})")
    return {cam: encode_image(images[cam], fmt=fmt, quality=quality,
                              max_smallest_side=max_smallest_side) for cam in CAMERAS}


def decode_observation(images: Dict[str, Dict[str, str]]) -> Dict[str, np.ndarray]:
    """The `images` field of an ActRequest -> {camera: HxWx3 uint8}."""
    out = {}
    for cam in CAMERAS:
        if cam not in images:
            raise ContractError(f"missing camera '{cam}'")
        out[cam] = decode_image(images[cam])
    return out


def payload_bytes(images: Dict[str, Dict[str, str]]) -> int:
    """Size of the encoded images, for logging/telemetry (base64 chars ~= bytes on the wire)."""
    return sum(len(v.get("data", "")) for v in images.values())


def describe(images: Dict[str, Any]) -> str:
    """One-line payload summary for logs, e.g. 'jpeg 3 cams 176.4 KiB'."""
    fmts = {v.get("format", "?") for v in images.values()}
    return f"{'/'.join(sorted(fmts))} {len(images)} cams {payload_bytes(images)/1024:.1f} KiB"
