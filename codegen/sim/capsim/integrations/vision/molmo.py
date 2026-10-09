"""Client for a Molmo vLLM server: point grounding from an image + text query."""
from __future__ import annotations

import base64
import io
import os
import re
import time
from collections.abc import Callable
from typing import Any

import PIL
import requests

_PROC: Any | None = None  # unused with the vLLM HTTP API
_MODEL: Any | None = None  # unused with the vLLM HTTP API

SERVICE_URL = os.environ.get("MOLMO_SERVICE_URL", "http://127.0.0.1:8122/v1")  # local



def _parse_points(text: str) -> tuple[list[tuple[float, float]], float]:
    """Parse (points, norm_scale) from Molmo output: Molmo2 coords (0-1000), else Molmo1/legacy/plain pairs (0-100)."""
    # Molmo2: <points coords="type obj_idx x y ...">: a type indicator, then (obj_idx, x, y) triplets.
    coords_match = re.search(r'<points\s+coords\s*=\s*["\']([^"\']+)["\']', text, flags=re.IGNORECASE)
    if coords_match:
        nums = [float(n) for n in coords_match.group(1).split()]
        points: list[tuple[float, float]] = []
        i = 1
        while i + 2 <= len(nums):
            x, y = nums[i + 1], nums[i + 2]
            points.append((x, y))
            i += 3
        return points, 1000.0

    # Molmo1: <point x=".." y="..">
    point_tags = re.findall(r"<point\b[^>]*>", text, flags=re.IGNORECASE)
    points = []
    for tag in point_tags:
        mx = re.search(r"\bx\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", tag, flags=re.IGNORECASE)
        my = re.search(r"\by\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", tag, flags=re.IGNORECASE)
        if mx and my:
            points.append((float(mx.group(1)), float(my.group(1))))
    if points:
        return [(x, y) for x, y in points if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0], 100.0

    # Legacy: <points x1=".." y1=".." ...>
    tag_match = re.search(r"<points\b[^>]*>", text, flags=re.IGNORECASE)
    if tag_match:
        source = tag_match.group(0)
        xs = {
            int(i): float(v)
            for i, v in re.findall(r"x(\d+)\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", source)
        }
        ys = {
            int(i): float(v)
            for i, v in re.findall(r"y(\d+)\s*=\s*['\"]([0-9]*\.?[0-9]+)['\"]", source)
        }
        idxs = sorted(set(xs) & set(ys))
        points = [(xs[i], ys[i]) for i in idxs]
        if points:
            return [(x, y) for x, y in points if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0], 100.0

    # Fallback: plain "x, y" pairs
    pairs = re.findall(r"([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)", text)
    points = [(float(x), float(y)) for x, y in pairs]
    return [(x, y) for x, y in points if 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0], 100.0


def convert_to_pixel_coordinates(
    points: list[tuple[float, float]], image: PIL.Image.Image, norm_scale: float = 100.0
) -> list[tuple[int, int]]:
    """Scale normalized (x, y) points (0..norm_scale) to pixel coordinates of ``image``."""
    width, height = image.size
    xs = [int(x / norm_scale * width) for x, _ in points]
    ys = [int(y / norm_scale * height) for _, y in points]
    return list(zip(xs, ys, strict=True))


def _image_to_data_url(image: PIL.Image.Image) -> str:
    """Encode a PIL image as a PNG data URL suitable for OpenAI-compatible APIs."""
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def init_molmo(
    model_name: str = "allenai/Molmo2-8B",
    base_url: str = SERVICE_URL,
    api_key: str | None = None,
) -> Callable[[PIL.Image.Image, list[str] | None], dict[str, tuple[int | None, int | None]]]:
    """Return ``det_fn(image, objects) -> {object: (x, y) pixel or (None, None)}`` backed by a vLLM server."""
    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    session = requests.Session()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def det_fn(
        image: PIL.Image.Image, objects: list[str] | None = None
    ) -> dict[str, tuple[int | None, int | None]]:
        """First pixel point per object query; (None, None) when nothing parsed."""

        if not objects:
            return {}

        img_url = _image_to_data_url(image)
        all_points: dict[str, tuple[int | None, int | None]] = {}
        for obj in objects:
            prompt = (
                f"Point at {obj}"
            )
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": img_url}},
                        ],
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.0,
                "stop": ["<|endoftext|>"],
            }

            max_retries, backoff = 3, 1.0
            for attempt in range(max_retries):
                try:
                    resp = session.post(chat_url, json=payload, headers=headers, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()
                    generated_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    print(f"Generated text for '{obj}': {generated_text}")
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"Request failed for '{obj}' (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(backoff * (2 ** attempt))
                    else:
                        generated_text = ""

            points, norm_scale = _parse_points(generated_text or "")
            if points:
                abs_coords = convert_to_pixel_coordinates(points, image, norm_scale)[0]
            else:
                abs_coords = (None, None)
                print("No points parsed from model output; skipping visualization.")
            all_points[obj] = abs_coords

        return all_points

    return det_fn


def init_molmo_all_points(
    model_name: str = "allenai/Molmo2-8B",
    base_url: str = SERVICE_URL,
    api_key: str | None = None,
) -> Callable[[PIL.Image.Image, str], list[tuple[int, int]]]:
    """Like init_molmo, but ``det_all_fn(image, query)`` returns every instance's pixel point ([] if none)."""
    chat_url = f"{base_url.rstrip('/')}/chat/completions"
    session = requests.Session()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def det_all_fn(image: PIL.Image.Image, query: str) -> list[tuple[int, int]]:
        img_url = _image_to_data_url(image)
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Point at {query}"},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.0,
            "stop": ["<|endoftext|>"],
        }
        max_retries, backoff = 3, 1.0
        generated_text = ""
        for attempt in range(max_retries):
            try:
                resp = session.post(chat_url, json=payload, headers=headers, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                generated_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"Generated text for '{query}': {generated_text}")
                break
            except Exception as e:  # noqa: BLE001
                print(f"Request failed for '{query}' (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(backoff * (2 ** attempt))

        points, norm_scale = _parse_points(generated_text or "")
        if not points:
            return []
        return convert_to_pixel_coordinates(points, image, norm_scale)

    return det_all_fn