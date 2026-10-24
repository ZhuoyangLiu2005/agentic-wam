#!/usr/bin/env python
"""Detect the camera resolution actually arriving at the server; refuse a mismatch with a clear error.

The Sharpa head-camera crop (sharpa_obs.HEAD_CROP = rows 0:280, cols 140:540) is in PIXELS of the
360x640 head_left frame the training shards were cut from. A frame of any other size shifts that
window silently -- a 720x1280 frame would keep only the top-left quarter of the training view, a
640x360 (H/W swapped) frame a rotated strip -- and nothing downstream notices: the policy just sees
a scene framed differently from training. So the frame size is checked on every /act, BEFORE any
preprocessing, and a mismatch is a ContractError (HTTP 400 from deploy/remote/server.py naming the
camera, the size it got and what it expected) rather than an AssertionError deep in inference
(HTTP 500) or, worse, a policy that runs on the wrong pixels.

No resizing or re-cropping is attempted: adapting a different resolution to the training crop is a
data-pipeline decision, not something to guess at the server.
"""
import logging
from math import gcd
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .contract import ContractError

logger = logging.getLogger(__name__)


def _aspect(h: int, w: int) -> str:
    g = gcd(w, h) or 1
    return f"{w // g}:{h // g}"


def _fmt(shape: Tuple[int, ...]) -> str:
    """Camera-style WxH, channels spelled out -- one convention for every message and /health."""
    if len(shape) == 3:
        return f"{shape[1]}x{shape[0]} (WxH) {shape[2]}ch"
    return f"array shape {shape}"


def diagnose(got: Tuple[int, ...], expected: Tuple[int, int, int]) -> str:
    """Name the most likely cause of a shape mismatch (best effort, for the error message)."""
    eh, ew, ec = expected
    if len(got) != 3:
        return f"expected an HxWx{ec} RGB array, got ndim={len(got)} -- grayscale or a flattened buffer?"
    h, w, c = got
    if c != ec:
        return {1: "single-channel (grayscale?)", 4: "4 channels (RGBA/BGRA?)"}.get(c, f"{c} channels") \
            + f" -- send {ec}-channel RGB"
    if (h, w) == (ew, eh):
        return "height and width are swapped (camera rotated, or the array was transposed)"
    if h * ew == w * eh:
        return (f"same {_aspect(eh, ew)} aspect but a different resolution -- the fixed training crop is "
                f"in {ew}x{eh} pixels, so set the camera driver to {ew}x{eh} (no server-side rescale)")
    return f"different aspect ratio ({_aspect(h, w)} vs the training {_aspect(eh, ew)})"


class ResolutionGuard:
    """Per-camera exact-shape check, plus what was last seen (for /health)."""

    def __init__(self, expected: Dict[str, Tuple[int, int, int]], label: str):
        self.expected = dict(expected)
        self.label = label
        self.reset()

    def reset(self) -> None:
        """Forget what was seen -- the startup warmup's synthetic frame is not an observation."""
        self.last_seen: Dict[str, Optional[Tuple[int, ...]]] = {cam: None for cam in self.expected}
        self.n_checked = 0
        self.n_rejected = 0
        self._logged_ok = False

    def check(self, images: Dict[str, Any]) -> None:
        """Raise ContractError if any expected camera's frame is not exactly its training size."""
        self.n_checked += 1
        for cam, exp in self.expected.items():
            if cam not in images:
                continue                       # presence is contract.ActRequest's job
            got = tuple(np.asarray(images[cam]).shape)
            self.last_seen[cam] = got
            if got != exp:
                self.n_rejected += 1
                raise ContractError(
                    f"images['{cam}']: got {_fmt(got)}, but the {self.label} checkpoint needs exactly "
                    f"{_fmt(exp)} -- {diagnose(got, exp)}")
        if not self._logged_ok:
            self._logged_ok = True
            logger.info("camera resolution OK (%s): %s", self.label,
                        ", ".join(f"{cam}={_fmt(s)}" for cam, s in self.last_seen.items() if s))

    def info(self) -> Dict[str, Any]:
        return {"expected": {cam: _fmt(s) for cam, s in self.expected.items()},
                "last_seen": {cam: (_fmt(s) if s else None) for cam, s in self.last_seen.items()},
                "n_checked": self.n_checked, "n_rejected": self.n_rejected}


def sharpa_guard() -> ResolutionGuard:
    """All three vega cameras: the converter asserted every source stream was 360x640x3."""
    from .sharpa_obs import RAW_SHAPE
    return ResolutionGuard({"head_camera": RAW_SHAPE, "left_camera": RAW_SHAPE, "right_camera": RAW_SHAPE},
                           label="Sharpa")
