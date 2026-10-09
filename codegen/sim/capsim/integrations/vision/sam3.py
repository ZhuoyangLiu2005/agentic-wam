"""Client for the SAM3 segmentation server (capsim/serving/launch_sam3_server.py)."""
from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from capsim.utils.serve_utils import post_with_retries

"""SAM 3 integration via FastAPI service."""

import os

# Env override allows several server stacks per host.
SERVICE_URL = os.environ.get("SAM3_SERVICE_URL", "http://127.0.0.1:8114")


def _encode_image(image: np.ndarray | Image.Image) -> str:
    if isinstance(image, np.ndarray):
        image_u8 = np.clip(image, 0, 255).astype(np.uint8) if image.dtype != np.uint8 else image
        pil_image = Image.fromarray(image_u8).convert("RGB")
    else:
        pil_image = image

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _decode_mask(mask_b64: str, shape: tuple[int, ...], dtype=np.uint8) -> np.ndarray:
    mask_bytes = base64.b64decode(mask_b64)
    return np.frombuffer(mask_bytes, dtype=dtype).reshape(shape)


def init_sam3() -> Any:
    """Return ``segment_fn(image, text_prompt) -> [{"mask", "box", "score", "label"}]`` for the SAM3 server."""

    def segment_fn(
        image: np.ndarray | Image.Image,
        text_prompt: str,
        box_prompt: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Run SAM3 text-prompt segmentation; [] on service failure."""
        encoded_image = _encode_image(image)
        payload = {"image_base64": encoded_image, "text_prompt": text_prompt}

        try:
            resp = post_with_retries(f"{SERVICE_URL}/segment", payload)
            results_data = resp["results"]
        except Exception as e:
            print(f"Failed to communicate with SAM3 service at {SERVICE_URL}: {e}")
            return []

        if not results_data:
            print(f"SAM3 returned no results for prompt: '{text_prompt}'")
            return []

        results = []
        for item in results_data:
            mask_shape = tuple(item["shape"])
            mask = _decode_mask(item["mask_base64"], mask_shape, dtype=np.uint8).astype(bool)

            results.append(
                {"mask": mask, "box": item["box"], "score": item["score"], "label": item["label"]}
            )

        return results

    return segment_fn


def init_sam3_point_prompt() -> Any:
    """Return ``point_prompt_fn(image, point_coords) -> [{"mask", "score"}]`` for the SAM3 server."""

    def point_prompt_fn(
        image: np.ndarray | Image.Image, point_coords: tuple[float, float]
    ) -> tuple[list[float], np.ndarray]:
        encoded_image = _encode_image(image)
        payload = {"image_base64": encoded_image, "point_coords": point_coords}

        try:
            resp = post_with_retries(f"{SERVICE_URL}/segment_point", payload)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to communicate with SAM3 service at {SERVICE_URL}: {e}")

        scores = resp.get("scores", [])
        masks_shape = tuple(resp.get("masks_shape", (0, 0, 0)))
        masks_dtype = np.dtype(resp.get("masks_dtype", "float32"))
        masks = _decode_mask(resp.get("masks_base64", ""), masks_shape, dtype=masks_dtype)
        masks = masks.astype(bool)
        results = [{"mask": mask, "score": score} for mask, score in zip(masks, scores)]

        return results

    return point_prompt_fn
