"""What the model actually sees — replicated locally so a human can check it.

The server applies the training-identical transform to every frame it receives:

    ResizeSmallestSideAspectPreserving(320x224)  ->  CenterCrop(320x224)  ->  [-1,1]

`ResizeSmallestSideAspectPreserving` scales so **both** dimensions are at least
the target, then the crop takes the middle. That means the *aspect ratio* of what
you send decides what gets cropped away, and nothing downstream will tell you if
it is wrong -- the model just quietly sees a differently-framed scene than it
trained on.

Worked through for the two cases that matter:

    station 640x480 (4:3)   scale max(320/640, 224/480) = 0.500  -> 320x240
                            crop 320x224  ->  8 px off top and bottom
    training rig 320x240    scale max(320/320, 224/240) = 1.000  -> 320x240
                            crop 320x224  ->  8 px off top and bottom   IDENTICAL

So the station's native 640x480 is **framing-identical** to the 320x240 rig the
model trained on. Send native and let the server resize -- one source of truth.

The trap, and why this module validates rather than assumes:

    640x360 (16:9)          scale max(320/640, 224/360) = 0.622  -> 398x224
                            crop 320x224  ->  39 px off EACH SIDE

A 16:9 camera loses a fifth of its horizontal field of view to the crop and
gains nothing vertically. The arms would be at the frame edge or outside it. No
error is raised anywhere; the policy just degrades. Hence `check_aspect()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_W, MODEL_H = 320, 224
"""What the model consumes, after the server's transform. Fixed by training
(`video_size=[224,320]` in deploy_policy.py) -- not a knob."""

TRAINING_ASPECT = 4.0 / 3.0
"""The training rig was 320x240. Anything with this aspect ratio crops
identically; anything else does not."""


@dataclass
class AspectReport:
    width: int
    height: int
    aspect: float
    matches_training: bool
    scaled_w: int
    scaled_h: int
    crop_x: int
    """Pixels removed from EACH side horizontally."""
    crop_y: int
    """Pixels removed from EACH side vertically."""

    def describe(self) -> str:
        frame = f"{self.width}x{self.height} (aspect {self.aspect:.3f})"
        step = f"-> resize {self.scaled_w}x{self.scaled_h} -> crop {MODEL_W}x{MODEL_H}"
        loss = f"drops {self.crop_x}px each side, {self.crop_y}px top/bottom"
        return f"{frame} {step}, {loss}"


def check_aspect(width: int, height: int) -> AspectReport:
    """Work out exactly what the server's transform will do to this frame size."""
    scale = max(MODEL_W / width, MODEL_H / height)
    scaled_w, scaled_h = round(width * scale), round(height * scale)
    aspect = width / height
    return AspectReport(
        width=width,
        height=height,
        aspect=aspect,
        matches_training=abs(aspect - TRAINING_ASPECT) < 0.02,
        scaled_w=scaled_w,
        scaled_h=scaled_h,
        crop_x=max(0, (scaled_w - MODEL_W) // 2),
        crop_y=max(0, (scaled_h - MODEL_H) // 2),
    )


def model_view(frame: np.ndarray) -> np.ndarray:
    """Return the 320x224 RGB crop the model will actually receive.

    Byte-for-byte intent-identical to the server transform, minus the final
    [-1,1] normalisation (which is not visually inspectable). Use this to *look*
    at what you are about to feed the policy.
    """
    height, width = frame.shape[:2]
    report = check_aspect(width, height)
    resized = cv2.resize(frame, (report.scaled_w, report.scaled_h), interpolation=cv2.INTER_LINEAR)
    y0 = (report.scaled_h - MODEL_H) // 2
    x0 = (report.scaled_w - MODEL_W) // 2
    return resized[y0 : y0 + MODEL_H, x0 : x0 + MODEL_W]


def preview_grid(frames: dict[str, np.ndarray], labels: bool = True) -> np.ndarray:
    """Side-by-side model-view of every camera, in the order the policy sees them.

    Save this before an eval and look at it. It is the only way to catch a
    swapped camera, a knocked-out viewpoint, a lens cap, or a wrong crop --
    `docs/REALWORLD.md` §4.3 calls viewpoint mismatch the single biggest
    real-world OOD risk, and nothing in software detects it.
    """
    order = ["head_camera", "left_camera", "right_camera"]
    tiles = []
    for name in order:
        if name not in frames:
            continue
        view = model_view(frames[name])
        tile = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        if labels:
            cv2.rectangle(tile, (0, 0), (MODEL_W - 1, 18), (0, 0, 0), -1)
            cv2.putText(
                tile, name, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA
            )
        tiles.append(tile)
    if not tiles:
        raise ValueError("no frames to preview")
    return np.hstack(tiles)


def save_preview(frames: dict[str, np.ndarray], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), preview_grid(frames))
    return path


def validate_cameras(frames: dict[str, np.ndarray]) -> list[str]:
    """Cheap sanity checks on the frames about to be sent. Returns warnings.

    None of these are conclusive -- a human looking at the preview is the real
    check -- but they catch the failures that are embarrassing to discover from
    a failed episode.
    """
    warnings: list[str] = []
    for name, frame in frames.items():
        height, width = frame.shape[:2]
        report = check_aspect(width, height)
        if not report.matches_training:
            warnings.append(
                f"{name}: {report.describe()} -- aspect differs from the 4:3 training rig, "
                f"so the crop keeps a different field of view than the model trained on"
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if float(gray.std()) < 5.0:
            warnings.append(f"{name}: nearly uniform image (std {gray.std():.1f}) -- lens cap, or a dead feed?")
        elif float(gray.mean()) < 15.0:
            warnings.append(f"{name}: very dark (mean {gray.mean():.1f}) -- check lighting/exposure")
    return warnings
