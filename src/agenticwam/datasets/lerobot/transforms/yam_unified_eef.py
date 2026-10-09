"""YAM/gripper raw-82D <-> unified-64D delta-EEF action/state transform.

Unified 64-D = 2 x [ 9-D EEF (pos3 + rot6d) | 22-D hand | 1-D gripper ]  (Left block then Right block).
YAM has no dexterous hand -> the 22-D hand block is ZERO. STATE = absolute EEF; ACTION = delta-base EEF in
the chunk-start (current) frame + ABSOLUTE gripper target + zero hand (delta arm, absolute hand).

Plugs into AgenticWAMProcessor.action_state_transforms (applied BEFORE normalize/merge; backward reverses it):
  forward:  raw 82-D  ->  unified 64-D (state abs, action delta)
  backward: unified 64-D delta action -> absolute EEF target (for deploy rollout), given the abs-EEF state.

Raw YAM 82-D layout (observation.state / action): ee_pose_L[0:16] (flat 4x4), ee_pose_R[16:32],
gripper_L[32], gripper_R[33] (joints[34:46] etc. unused — we move to EEF control).
"""
import torch

EEF_L = slice(0, 16); EEF_R = slice(16, 32); GRIP_L = 32; GRIP_R = 33
HAND = 22
# 64-D block offsets:  L: eef 0:9, hand 9:31, grip 31:32 | R: eef 32:41, hand 41:63, grip 63:64


def _pose16_to_Rt(p):
    M = p.reshape(*p.shape[:-1], 4, 4)
    return M[..., :3, :3], M[..., :3, 3]

def _Rt_to_9d(R, t):
    return torch.cat([t, R[..., :, 0], R[..., :, 1]], dim=-1)          # [...,9] = t(3)+col0(3)+col1(3)

def _rot6d_to_R(d6):
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 / (a2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)                           # columns

def _mm(A, B): return torch.einsum("...ij,...jk->...ik", A, B)
def _mv(A, v): return torch.einsum("...ij,...j->...i", A, v)


class YamUnifiedEEFTransform:
    def __init__(self, keys):
        self.keys = list(keys)

    def _state64(self, s):                                              # s:[...,82] -> [...,64]
        RL, tL = _pose16_to_Rt(s[..., EEF_L]); RR, tR = _pose16_to_Rt(s[..., EEF_R])
        z = torch.zeros(*s.shape[:-1], HAND, dtype=s.dtype, device=s.device)
        return torch.cat([_Rt_to_9d(RL, tL), z, s[..., GRIP_L:GRIP_L+1],
                          _Rt_to_9d(RR, tR), z, s[..., GRIP_R:GRIP_R+1]], dim=-1)

    def _action64(self, a, base):                                      # a:[...,Tc,82], base:[...,82] -> [...,Tc,64]
        RbL, tbL = _pose16_to_Rt(base[..., EEF_L]); RbR, tbR = _pose16_to_Rt(base[..., EEF_R])
        RbL, tbL = RbL.unsqueeze(-3), tbL.unsqueeze(-2); RbR, tbR = RbR.unsqueeze(-3), tbR.unsqueeze(-2)
        RaL, taL = _pose16_to_Rt(a[..., EEF_L]); RaR, taR = _pose16_to_Rt(a[..., EEF_R])
        dL = torch.cat([_mv(RbL.transpose(-1,-2), taL - tbL), _mm(RbL.transpose(-1,-2), RaL)[..., :, 0],
                        _mm(RbL.transpose(-1,-2), RaL)[..., :, 1]], dim=-1)
        dR = torch.cat([_mv(RbR.transpose(-1,-2), taR - tbR), _mm(RbR.transpose(-1,-2), RaR)[..., :, 0],
                        _mm(RbR.transpose(-1,-2), RaR)[..., :, 1]], dim=-1)
        z = torch.zeros(*a.shape[:-1], HAND, dtype=a.dtype, device=a.device)
        return torch.cat([dL, z, a[..., GRIP_L:GRIP_L+1], dR, z, a[..., GRIP_R:GRIP_R+1]], dim=-1)

    def forward(self, batch):
        for k in self.keys:
            s = batch["state"][k]
            if "action" in batch:                                      # base = chunk-start (CURRENT) frame = s[0]
                # FIX 2026-07-17: was s[...,-1,:] (LAST/future frame) -> delta anchored to future,
                # mismatching the eval decode (current pose). Correct: delta_k = a_k - s_t = s[...,0,:].
                batch["action"][k] = self._action64(batch["action"][k], s[..., 0, :])
            batch["state"][k] = self._state64(s)
        return batch

    def backward(self, batch):
        # unified-64D delta action -> absolute EEF (9-D) target, using the abs-EEF state as base.
        for k in self.keys:
            st = batch["state"][k]                                     # [...,Tobs,64] abs
            act = batch["action"][k]                                   # [...,Tc,64] delta
            bL, bR = st[..., 0, 0:9], st[..., 0, 32:41]                # base abs eef 9d (L,R) = current (chunk-start); FIX 2026-07-17 was -1
            RbL = _rot6d_to_R(bL[..., 3:9]); tbL = bL[..., 0:3]
            RbR = _rot6d_to_R(bR[..., 3:9]); tbR = bR[..., 0:3]
            def compose(d9, Rb, tb):
                Rd = _rot6d_to_R(d9[..., 3:9]); Ra = _mm(Rb.unsqueeze(-3), Rd)
                ta = tb.unsqueeze(-2) + _mv(Rb.unsqueeze(-3), d9[..., 0:3])
                return torch.cat([ta, Ra[..., :, 0], Ra[..., :, 1]], dim=-1)
            aL = compose(act[..., 0:9], RbL, tbL); aR = compose(act[..., 32:41], RbR, tbR)
            batch["action"][k] = torch.cat([aL, act[..., 9:31], act[..., 31:32],
                                            aR, act[..., 41:63], act[..., 63:64]], dim=-1)
        return batch
