#!/usr/bin/env python
"""14-D joint <-> 64-D unified-EEF bridge for the REAL YAM station.

The unified YAM checkpoints are trained in the 64-D
unified EEF space (YamUnifiedEEFTransform): STATE = absolute EEF (pos3+rot6d)
+ absolute gripper; ACTION = delta EEF anchored to the CHUNK-START state +
ABSOLUTE gripper; the 22-D hand block is zero. The robot client speaks 14-D
absolute joints and nothing else (deploy/robot/agenticwam_client.py refuses
action_dim != 14). This module closes that gap ON THE SERVER:

    in:   state14  --FK-->  raw 82-D  (what AgenticWAMProcessor's transform expects)
    out:  delta-64 chunk  --apply_delta_9d + damped-LS IK-->  absolute 14-D joints

Kinematics come from utils/yam_fk_ik/yam_kinematics.py, validated against the
real station (FK matches the recording exactly; IK round-trip ~mrad --
utils/yam_fk_ik/REPLAY_GUIDE.md).

Decode anchors EVERY row to the chunk-start EEF pose -- the exact inverse of
YamUnifiedEEFTransform._action64 (base = s[..., 0, :]) -- NOT a chained
row-to-row composition. IK is seeded with the previous row's solution so the
solver tracks the trajectory and each solve converges in a few iterations.

Thread safety: yam_kinematics keeps one global MjData; callers must hold a
lock across decode_chunk (PolicyRuntime serializes inference anyway).
"""
import sys
from pathlib import Path

import numpy as np

_YAM_FK_IK = str(Path(__file__).resolve().parents[2] / "utils" / "yam_fk_ik")
if _YAM_FK_IK not in sys.path:
    sys.path.insert(0, _YAM_FK_IK)

import yam_kinematics as yk  # noqa: E402  (loads the mujoco station model once)

L_EEF = slice(0, 9)
L_GRIP = 31
R_EEF = slice(32, 41)
R_GRIP = 63


def state14_to_82(state14: np.ndarray) -> np.ndarray:
    """Robot joints -> the raw 82-D layout the training processor expects."""
    return yk.state14_to_82(np.asarray(state14, np.float64))


def joints_to_delta64(rows14: np.ndarray, anchor14: np.ndarray) -> np.ndarray:
    """[K,14] absolute joints -> [K,64] unified delta actions anchored at
    `anchor14` (the request's state). Exact inverse direction of decode_chunk;
    used to express already-committed joint actions in the model's action
    space for RTC inpainting guidance."""
    a = np.asarray(anchor14, np.float64).reshape(-1)
    TbL, TbR = yk.fk_both(a[0:6], a[7:13])
    rows = np.asarray(rows14, np.float64)
    out = np.zeros((len(rows), 64))
    for i, r in enumerate(rows):
        TL, TR = yk.fk_both(r[0:6], r[7:13])
        out[i, 0:9] = yk.eef_delta_9d(TbL, TL)
        out[i, L_GRIP] = r[6]
        out[i, 32:41] = yk.eef_delta_9d(TbR, TR)
        out[i, R_GRIP] = r[13]
    return out


def decode_chunk(delta64: np.ndarray, state14: np.ndarray):
    """[T,64] unified delta chunk + current 14-D state -> ([T,14] ABSOLUTE joints, ik_failures).

    Grippers are absolute in the unified layout and pass through unchanged.
    A row whose IK does not reach the 1 mm/1 mrad tolerance still uses its
    (step-clipped) solution -- the station's safety envelope rate-limits every
    step against the measured position regardless -- but it is counted, and the
    caller should surface the count: a chunk full of IK failures is the model
    asking for unreachable poses, which is a finding, not noise.
    """
    d = np.asarray(delta64, np.float64)
    s = np.asarray(state14, np.float64).reshape(-1)
    T_L, T_R = yk.fk_both(s[0:6], s[7:13])   # chunk-start anchor, fixed for all rows
    qL, qR = s[0:6].copy(), s[7:13].copy()   # IK seed walks with the trajectory
    out = np.zeros((d.shape[0], 14))
    failures = 0
    for t in range(d.shape[0]):
        ok_l, qL = yk.ik(yk.apply_delta_9d(T_L, d[t, L_EEF]), "left", init6=qL)
        ok_r, qR = yk.ik(yk.apply_delta_9d(T_R, d[t, R_EEF]), "right", init6=qR)
        failures += (not ok_l) + (not ok_r)
        out[t, 0:6] = qL
        out[t, 6] = d[t, L_GRIP]
        out[t, 7:13] = qR
        out[t, 13] = d[t, R_GRIP]
    return out, failures
