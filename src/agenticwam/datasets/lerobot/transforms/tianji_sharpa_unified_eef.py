"""Tianji+Sharpa raw-76D <-> unified-64D delta-EEF transform.

Unified 64-D = 2 x [ 9-D EEF (pos3 + rot6d) | 22-D hand | 1-D gripper ]  (Left block, then Right).
Tianji carries a Sharpa dexterous hand -> the 22-D hand block is POPULATED and the 1-D gripper
block is ZERO. That is the exact mirror of YAM (gripper live, hand zero), and it is why one 64-D
action head can serve both embodiments.

RAW 76-D LAYOUT (produced by the tianji/Sharpa raw-dump converter, NOT a vendor format):
    [ 0:16]  L arm pose, 4x4 homogeneous, ROW-MAJOR flattened   <- action_tianji/left_arm_pose
    [16:32]  R arm pose, 4x4 homogeneous, ROW-MAJOR flattened   <- action_tianji/right_arm_pose
    [32:54]  L hand, 22 joint angles, RADIANS                   <- action_left_hand/joint_positions
    [54:76]  R hand, 22 joint angles, RADIANS                   <- action_right_hand/joint_positions

ROTATION PACKING. The source stores a full 4x4 matrix, so
there is NO row-vs-column ambiguity to resolve: R is recovered unambiguously as M[:3,:3] and then
re-encoded as the first two COLUMNS, matching yam_unified_eef.py and the deploy-side decoder.

ACTION SEMANTICS (verified on the raw dumps):
  * arm pose is ABSOLUTE (R orthonormal to 1.8e-15, bottom row [0,0,0,1], translation in metres)
  * hand joints are ABSOLUTE radians (observed range -0.117 .. 1.905, inside Sharpa URDF limits)
  * action vs state differ by the teleop command-vs-measured gap (0.14 pose / 0.051 rad), i.e.
    `action` is the commanded target and `state` the achieved one -- both absolute.
This transform converts to the contract shared with YAM and ego:
    STATE  = absolute EEF + absolute hand
    ACTION = delta-base EEF in the CHUNK-START frame + ABSOLUTE hand + ZERO gripper
"""
from __future__ import annotations

import torch

POSE_L = slice(0, 16)
POSE_R = slice(16, 32)
HAND_L = slice(32, 54)
HAND_R = slice(54, 76)
RAW_DIM = 76
# 64-D block offsets: L eef 0:9, hand 9:31, grip 31:32 | R eef 32:41, hand 41:63, grip 63:64


def _pose16_to_Rt(p):
    """[...,16] row-major 4x4 -> (R[...,3,3], t[...,3])."""
    M = p.reshape(*p.shape[:-1], 4, 4)
    return M[..., :3, :3], M[..., :3, 3]


def _Rt_to_9d(R, t):
    """(R, t) -> [...,9] = t(3) + col0(3) + col1(3).  COLUMNS -- matches yam_unified_eef."""
    return torch.cat([t, R[..., :, 0], R[..., :, 1]], dim=-1)


def _mm(A, B):
    return torch.einsum("...ij,...jk->...ik", A, B)


def _mv(A, v):
    return torch.einsum("...ij,...j->...i", A, v)


def _rot6d_to_R(d6):
    """[...,6] (two columns, Gram-Schmidt) -> R[...,3,3]. Inverse of the packing in _Rt_to_9d."""
    eps = torch.finfo(d6.dtype).eps
    a0, a1 = d6[..., 0:3], d6[..., 3:6]
    b0 = a0 / a0.norm(dim=-1, keepdim=True).clamp_min(eps)
    a1 = a1 - (b0 * a1).sum(-1, keepdim=True) * b0
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(eps)
    b2 = torch.cross(b0, b1, dim=-1)
    return torch.stack([b0, b1, b2], dim=-1)  # stack as COLUMNS


class TianjiSharpaUnifiedEEFTransform:
    """raw 76-D -> unified 64-D. Plugs into AgenticWAMProcessor.action_state_transforms."""

    def __init__(self, keys):
        self.keys = list(keys)

    # ---- forward ------------------------------------------------------------------------
    def _state64(self, s):                                    # [...,76] -> [...,64]
        RL, tL = _pose16_to_Rt(s[..., POSE_L])
        RR, tR = _pose16_to_Rt(s[..., POSE_R])
        g = torch.zeros(*s.shape[:-1], 1, dtype=s.dtype, device=s.device)   # Sharpa: no gripper
        return torch.cat([_Rt_to_9d(RL, tL), s[..., HAND_L], g,
                          _Rt_to_9d(RR, tR), s[..., HAND_R], g], dim=-1)

    def _action64(self, a, base):        # a:[...,Tc,76], base:[...,76] -> [...,Tc,64]
        RbL, tbL = _pose16_to_Rt(base[..., POSE_L])
        RbR, tbR = _pose16_to_Rt(base[..., POSE_R])
        RbL, tbL = RbL.unsqueeze(-3), tbL.unsqueeze(-2)
        RbR, tbR = RbR.unsqueeze(-3), tbR.unsqueeze(-2)
        RaL, taL = _pose16_to_Rt(a[..., POSE_L])
        RaR, taR = _pose16_to_Rt(a[..., POSE_R])
        dRL = _mm(RbL.transpose(-1, -2), RaL)
        dRR = _mm(RbR.transpose(-1, -2), RaR)
        dL = torch.cat([_mv(RbL.transpose(-1, -2), taL - tbL), dRL[..., :, 0], dRL[..., :, 1]], -1)
        dR = torch.cat([_mv(RbR.transpose(-1, -2), taR - tbR), dRR[..., :, 0], dRR[..., :, 1]], -1)
        g = torch.zeros(*a.shape[:-1], 1, dtype=a.dtype, device=a.device)
        # hand blocks pass through UNCHANGED -> absolute, exactly like YAM's gripper.
        return torch.cat([dL, a[..., HAND_L], g, dR, a[..., HAND_R], g], dim=-1)

    # NOTE the batch API: AgenticWAMProcessor passes `batch["state"][key]` / `batch["action"][key]`
    # dicts, not bare tensors. This mirrors YamUnifiedEEFTransform exactly.
    def forward(self, batch):
        for k in self.keys:
            s = batch["state"][k]                                 # [...,Tobs,76] raw
            if "action" in batch:
                # base = chunk-start (CURRENT) frame = s[0], NOT s[-1]: the delta must be anchored
                # to the pose the policy actually observes at execution time.
                batch["action"][k] = self._action64(batch["action"][k], s[..., 0, :])
            batch["state"][k] = self._state64(s)
        return batch

    # ---- backward -----------------------------------------------------------------------
    def backward(self, batch):
        """unified 64-D delta action -> ABSOLUTE 64-D EEF target, given the abs-EEF 64-D state.

        Deliberately identical to YamUnifiedEEFTransform.backward -- both embodiments are
        column-packed, so deploy gets ONE shared inverse. The 22-D hand and the (zero) gripper are
        already absolute and pass through untouched.
        """
        for k in self.keys:
            st = batch["state"][k]                                # [...,Tobs,64] absolute
            act = batch["action"][k]                              # [...,Tc,64] delta
            bL, bR = st[..., 0, 0:9], st[..., 0, 32:41]           # base = chunk-start abs EEF
            RbL, tbL = _rot6d_to_R(bL[..., 3:9]), bL[..., 0:3]
            RbR, tbR = _rot6d_to_R(bR[..., 3:9]), bR[..., 0:3]

            def compose(d9, Rb, tb):
                Rd = _rot6d_to_R(d9[..., 3:9])
                Ra = _mm(Rb.unsqueeze(-3), Rd)
                ta = tb.unsqueeze(-2) + _mv(Rb.unsqueeze(-3), d9[..., 0:3])
                return torch.cat([ta, Ra[..., :, 0], Ra[..., :, 1]], dim=-1)

            aL = compose(act[..., 0:9], RbL, tbL)
            aR = compose(act[..., 32:41], RbR, tbR)
            batch["action"][k] = torch.cat([aL, act[..., 9:31], act[..., 31:32],
                                            aR, act[..., 41:63], act[..., 63:64]], dim=-1)
        return batch
