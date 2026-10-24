#!/usr/bin/env python
"""Sharpa/vega camera preprocessing -- byte-identical to training.

utils/convert_trex_v30_to_v21.py built the Sharpa training shards with ONE crop recipe per
camera, not the generic ResizeSmallestSideAspectPreserving->CenterCrop used by the YAM deploy
path (deploy/robot/obs_pipeline.py / robotwin/deploy/deploy_policy.py):

    cam_high (head_left):        crop rows 0:280, cols 140:540 (280x400, same aspect as 320x224)
                                  -> cv2.resize to 320x224, INTER_AREA
    cam_left/right_wrist:        NO crop -- cv2.resize the full 360x640 frame straight to
                                  320x224 (a non-aspect-preserving squash), INTER_AREA

Both functions here reproduce that exactly. Applying them BEFORE a frame reaches
AgenticWAMCosmosRobotWinPolicy's generic resize+center-crop (which still runs downstream,
unconditionally, for every embodiment) makes that generic step a no-op: resizing an
already-320x224 image to "at least 320x224" is scale=1, and center-cropping 320x224 out of a
320x224 image is the identity crop.
"""
import cv2
import numpy as np

HEAD_CROP = (0, 280, 140, 540)   # rows y0:y1, cols x0:x1 on the 360x640 head_left frame
MODEL_W, MODEL_H = 320, 224
RAW_SHAPE = (360, 640, 3)


def head_camera_view(frame: np.ndarray) -> np.ndarray:
    """cam_high: fixed crop, then resize. Matches utils/convert_trex_v30_to_v21.py exactly."""
    assert frame.shape == RAW_SHAPE, f"expected raw {RAW_SHAPE} head frame, got {frame.shape}"
    y0, y1, x0, x1 = HEAD_CROP
    cropped = frame[y0:y1, x0:x1]
    return cv2.resize(cropped, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)


def wrist_camera_view(frame: np.ndarray) -> np.ndarray:
    """cam_left/right_wrist: no crop, squash-resize the full frame."""
    assert frame.shape == RAW_SHAPE, f"expected raw {RAW_SHAPE} wrist frame, got {frame.shape}"
    return cv2.resize(frame, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
