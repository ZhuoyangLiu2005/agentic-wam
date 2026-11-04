#!/usr/bin/env python
"""14-D joint <-> 64-D unified-EEF bridge for the dual ARX X5 in RoboDojo (sim).

The RoboDojo checkpoints are post-trained on data converted by utils/convert_robodojo_v21_to_yam82.py: the YAM 82-D raw
layout with link6 poses from X5 FK in each arm's own base frame, so YamUnifiedEEFTransform applies unchanged (STATE =
absolute EEF pos3+rot6d + absolute gripper; ACTION = EEF delta anchored at the CHUNK-START state + absolute gripper).
This module is yam_eef64.py with the X5 kinematics (utils/x5_kinematics.py) in place of the YAM station model:

    in:   state14 [L q0..5, L grip, R q0..5, R grip]  --X5 FK-->  raw 82-D
    out:  delta-64 chunk  --apply_delta_9d + damped-LS IK-->  absolute 14-D joints

Grippers are RoboDojo's normalised ee joint state (1 = open, ~0.08 = closed); the sim's joint action takes the same
normalisation (src/eval_client/eval_env.py is the inverse of obs_manager.py), so they pass through.
"""
import sys
from pathlib import Path

import numpy as np

_UTILS = str(Path(__file__).resolve().parents[2] / "utils")
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)

import x5_kinematics as xk  # noqa: E402

L_EEF = slice(0, 9)
L_GRIP = 31
R_EEF = slice(32, 41)
R_GRIP = 63


def state14_to_82(state14: np.ndarray) -> np.ndarray:
    """RoboDojo joints -> the raw 82-D layout the training processor expects (same as the converter's pack82)."""
    s = np.asarray(state14, np.float64).reshape(-1)
    o = np.zeros(82, np.float32)
    o[0:16] = xk.fk(s[0:6]).reshape(16)
    o[16:32] = xk.fk(s[7:13]).reshape(16)
    o[32], o[33] = s[6], s[13]
    o[34:40], o[40:46] = s[0:6], s[7:13]
    return o


def joints_to_delta64(rows14: np.ndarray, anchor14: np.ndarray) -> np.ndarray:
    """[K,14] absolute joints -> [K,64] unified delta actions anchored at `anchor14` (inverse of decode_chunk)."""
    a = np.asarray(anchor14, np.float64).reshape(-1)
    TbL, TbR = xk.fk(a[0:6]), xk.fk(a[7:13])
    rows = np.asarray(rows14, np.float64)
    out = np.zeros((len(rows), 64))
    for i, r in enumerate(rows):
        out[i, L_EEF] = xk.eef_delta_9d(TbL, xk.fk(r[0:6]))
        out[i, L_GRIP] = r[6]
        out[i, R_EEF] = xk.eef_delta_9d(TbR, xk.fk(r[7:13]))
        out[i, R_GRIP] = r[13]
    return out


def decode_chunk(delta64: np.ndarray, state14: np.ndarray):
    """[T,64] unified delta chunk + current 14-D state -> ([T,14] ABSOLUTE joints, ik_failures).

    Every row is anchored at the chunk-start pose (the inverse of YamUnifiedEEFTransform._action64); IK is seeded
    with the previous row's solution. A row whose IK misses tolerance still uses its solution and is counted."""
    d = np.asarray(delta64, np.float64)
    s = np.asarray(state14, np.float64).reshape(-1)
    T_L, T_R = xk.fk(s[0:6]), xk.fk(s[7:13])
    qL, qR = s[0:6].copy(), s[7:13].copy()
    out = np.zeros((d.shape[0], 14))
    failures = 0
    for t in range(d.shape[0]):
        ok_l, qL = xk.ik(xk.apply_delta_9d(T_L, d[t, L_EEF]), qL)
        ok_r, qR = xk.ik(xk.apply_delta_9d(T_R, d[t, R_EEF]), qR)
        failures += (not ok_l) + (not ok_r)
        out[t, 0:6] = qL
        out[t, 6] = d[t, L_GRIP]
        out[t, 7:13] = qR
        out[t, 13] = d[t, R_GRIP]
    return out, failures
