#!/usr/bin/env python
"""58-D vega joints <-> 64-D unified-EEF bridge for the REAL Tianji+Sharpa station.

The Sharpa checkpoints are trained in the 64-D unified EEF space
(TianjiSharpaUnifiedEEFTransform): STATE = absolute EEF (pos3+rot6d) + absolute 22-D hand + zero
gripper per arm; ACTION = delta EEF anchored to the CHUNK-START state + ABSOLUTE hand + zero
gripper. The real vega robot's control layer is joint-space: 58-D
`[L_arm_j1..7 (7), L_hand (22), R_arm_j1..7 (7), R_hand (22)]` (matches the raw dims T-Rex's own
`unpack_lerobot_to_raw_fk.py` uses for `observation.state`/`action`). This module closes that gap
ON THE SERVER, mirroring deploy/common/yam_eef64.py:

    in:   state58  --FK (vega_kinematics)-->  raw 76-D  (TianjiSharpaUnifiedEEFTransform expects)
    out:  delta-64 chunk  --transform.backward + IK (vega_kinematics)-->  absolute 58-D joints

`TianjiSharpaUnifiedEEFTransform.backward` is imported and reused directly (not re-derived) so
this bridge can never silently drift from the training-time rot6d/delta convention -- only its
private `_state64`/`backward` are called, which is the exact mirror of what
`deploy/common/yam_eef64.py` does implicitly via `YamUnifiedEEFTransform`'s identical packing.

Hand joints are absolute in BOTH the raw-76 and unified-64 layout and need no kinematics at all;
only the two 7-DOF arms go through FK/IK (utils/vega_fk_ik/vega_kinematics.py).

A row whose IK misses the 1mm tolerance still returns its best (step-clipped) solution but is
counted -- same convention as yam_eef64.decode_chunk's `ik_failures`.
"""
import sys
from pathlib import Path

import numpy as np
import torch

_AGENTICWAM_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _AGENTICWAM_SRC not in sys.path:
    sys.path.insert(0, _AGENTICWAM_SRC)

from agenticwam.datasets.lerobot.transforms.tianji_sharpa_unified_eef import (  # noqa: E402
    TianjiSharpaUnifiedEEFTransform, _rot6d_to_R,
)

_VEGA_FK_IK = str(Path(__file__).resolve().parents[2] / "utils" / "vega_fk_ik")
if _VEGA_FK_IK not in sys.path:
    sys.path.insert(0, _VEGA_FK_IK)
import vega_kinematics as vk  # noqa: E402  (loads the vega_1 URDF once)

_TRANSFORM = TianjiSharpaUnifiedEEFTransform(keys=["default"])

L_ARM = slice(0, 7)
L_HAND = slice(7, 29)
R_ARM = slice(29, 36)
R_HAND = slice(36, 58)

L_EEF9 = slice(0, 9)
L_HAND64 = slice(9, 31)
R_EEF9 = slice(32, 41)
R_HAND64 = slice(41, 63)


def state58_to_raw76(state58: np.ndarray) -> np.ndarray:
    """Robot's 58-D joint/hand vector -> the raw-76-D layout the training processor expects."""
    s = np.asarray(state58, np.float64).reshape(-1)
    assert s.shape == (58,), f"expected [58] state, got {s.shape}"
    T_L, T_R = vk.fk_both(s[L_ARM], s[R_ARM])
    raw76 = np.zeros(76, np.float64)
    raw76[0:16] = T_L.reshape(-1)
    raw76[16:32] = T_R.reshape(-1)
    raw76[32:54] = s[L_HAND]
    raw76[54:76] = s[R_HAND]
    return raw76


def _pose_from_abs9(abs9: np.ndarray) -> np.ndarray:
    """[9] = t(3) + rot6d(6), absolute -> 4x4 homogeneous pose."""
    t = abs9[0:3]
    R = _rot6d_to_R(torch.as_tensor(abs9[3:9], dtype=torch.float32)).numpy()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def decode_chunk(delta64: np.ndarray, state58: np.ndarray):
    """[T,64] unified delta chunk + current 58-D state -> ([T,58] ABSOLUTE joints/hand, ik_failures).

    Hands are absolute in the unified layout and pass through unchanged. A row whose IK does not
    reach the 1mm tolerance still uses its (step-clipped) solution -- the station's safety
    envelope rate-limits every step against the measured position regardless -- but it is
    counted, and the caller should surface it: a chunk full of IK failures is the model asking
    for unreachable poses, which is a finding, not noise.
    """
    d = np.asarray(delta64, np.float64)
    s = np.asarray(state58, np.float64).reshape(-1)
    assert d.ndim == 2 and d.shape[1] == 64, f"expected [T,64] delta chunk, got {d.shape}"
    assert s.shape == (58,), f"expected [58] state, got {s.shape}"

    raw76 = state58_to_raw76(s)
    state64 = _TRANSFORM._state64(torch.as_tensor(raw76, dtype=torch.float32).reshape(1, 1, 76))
    delta64_t = torch.as_tensor(d, dtype=torch.float32).reshape(1, d.shape[0], 64)
    batch = {"state": {"default": state64}, "action": {"default": delta64_t}}
    abs64 = _TRANSFORM.backward(batch)["action"]["default"][0].numpy()  # [T,64]

    qL, qR = s[L_ARM].copy(), s[R_ARM].copy()   # IK seed walks with the trajectory
    out = np.zeros((d.shape[0], 58))
    failures = 0
    for t in range(d.shape[0]):
        T_L = _pose_from_abs9(abs64[t, L_EEF9])
        T_R = _pose_from_abs9(abs64[t, R_EEF9])
        ok, qL, qR = vk.ik_both(T_L, T_R, qL, qR)
        failures += int(not ok)
        out[t, L_ARM] = qL
        out[t, L_HAND] = abs64[t, L_HAND64]
        out[t, R_ARM] = qR
        out[t, R_HAND] = abs64[t, R_HAND64]
    return out, failures
