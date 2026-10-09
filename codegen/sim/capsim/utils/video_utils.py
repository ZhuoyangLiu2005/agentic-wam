"""Video encode/decode helpers (frames <-> mp4, base64 for LLM prompts)."""
import base64
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, TypeVar, Union

import cv2
import imageio
import numpy as np
from PIL import Image


def _encode_video_base64(frames: list[np.ndarray], fps: int = 30) -> str:
    """Encode RGB frames as an MP4 ``data:video/mp4;base64,...`` URL."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with imageio.get_writer(
            tmp_path, fps=fps, format="FFMPEG", codec="libx264"
        ) as writer:
            for frame in frames:
                writer.append_data(np.ascontiguousarray(frame))
        with open(tmp_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    finally:
        os.unlink(tmp_path)
    return f"data:video/mp4;base64,{b64}"


def _write_video(frames: list[np.ndarray], output_dir: str | None, *, suffix: str) -> None:
    """Write RGB frames to ``<output_dir>/video_<suffix>.mp4`` at 30 fps."""
    parent = Path(output_dir)
    out_path = parent / f"video_{suffix}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path, fps=30, format="FFMPEG", codec="libx264") as writer:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(frame))
    print(f"Saved interaction video to {out_path} ({len(frames)} frames)")


def resize_with_pad(
    images: np.ndarray,
    height: int,
    width: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize image(s) (h, w, c) or (b, h, w, c) to height x width without distortion, padding with black."""
    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images[None]  # Add batch dimension

    batch_size, cur_height, cur_width, channels = images.shape

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_images = np.zeros((batch_size, resized_height, resized_width, channels), dtype=images.dtype)

    for i in range(batch_size):
        resized_images[i] = cv2.resize(images[i], (resized_width, resized_height), interpolation=interpolation)

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    if images.dtype == np.uint8:
        pad_value = 0
    elif images.dtype == np.float32:
        pad_value = -1.0
    else:
        pad_value = 0

    padded_images = np.pad(
        resized_images,
        ((0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )

    if not has_batch_dim:
        padded_images = padded_images[0]

    return padded_images


def resize_with_center_crop(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicate tf.image.resize_with_center_crop for a batch of [..., h, w, c] images using PIL."""
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [np.array(_resize_with_center_crop(Image.fromarray(im), height, width, method=method)) for im in images]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_center_crop(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicate tf.image.resize_with_center_crop for one PIL image."""
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    # scale up to cover the target, then crop the excess
    ratio = max(height / cur_height, width / cur_width)
    resized_width = int(cur_width * ratio)
    resized_height = int(cur_height * ratio)

    resized_image = image.resize((resized_width, resized_height), resample=method)

    crop_w0 = (resized_width - width) // 2
    crop_h0 = (resized_height - height) // 2

    crop_w0 = max(0, crop_w0)
    crop_h0 = max(0, crop_h0)
    crop_w1 = min(resized_width, crop_w0 + width)
    crop_h1 = min(resized_height, crop_h0 + height)

    cropped_image = resized_image.crop((crop_w0, crop_h0, crop_w1, crop_h1))

    if cropped_image.size != (width, height):
        cropped_image = cropped_image.resize((width, height), resample=method)

    assert cropped_image.size == (width, height)
    return cropped_image