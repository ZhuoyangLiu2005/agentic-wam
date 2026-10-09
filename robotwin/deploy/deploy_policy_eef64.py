"""Eval support for 64-D unified-EEF RoboTwin models (agenticwam_cosmos). install() patches the base
agenticwam_cosmos deploy_policy IN PLACE so the sim (14-D Aloha qpos) round-trips through the 64-D
delta-EEF space:
  proprio:  sim 14-D joints --cuRobo FK--> 34-D raw dual-EEF --_state64--> 64-D  (feeds model)
  action:   model 64-D denorm delta --tf.backward--> abs dual-EEF --cuRobo IK--> 14-D qpos

Model-agnostic: identical to AgenticWAM's eef64 wrapper (same 64-D space); only the imports point at
the agenticwam_cosmos package + the shared RoboTwin cuRobo assets. Enable via env AGENTICWAM_ACTION_SPACE=eef64.
"""
import os, sys
import numpy as np, torch

# RoboTwin cuRobo assets (aloha-agilex) of the RoboTwin checkout that EVALUATION.robotwin_root points at.
RT = os.environ.get("AGENTICWAM_ROBOTWIN_ROOT", "/path/to/RoboTwin")   # = paths.robotwin_root
A = f"{RT}/assets/embodiments/aloha-agilex"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deploy_policy as base_dp
# agenticwam_cosmos's YamUnifiedEEFTransform (base_dp already put agenticwam_cosmos/src on sys.path).
from agenticwam.datasets.lerobot.transforms.yam_unified_eef import YamUnifiedEEFTransform, _rot6d_to_R

from curobo.util_file import load_yaml
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
_TDT = TensorDeviceType()


def _build_ik(yml):
    cfg = load_yaml(yml); rc = cfg.get("robot_cfg", cfg)
    return IKSolver(IKSolverConfig.load_from_robot_config(
        rc, None, tensor_args=_TDT, num_seeds=8, self_collision_check=False,
        self_collision_opt=False, use_cuda_graph=False))


def _R_from_9d(d9):
    return _rot6d_to_R(d9[..., 3:9])


def _quat_R(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]; B = q.shape[0]
    R = torch.empty(B, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1-2*(y*y+z*z); R[:, 0, 1] = 2*(x*y-w*z); R[:, 0, 2] = 2*(x*z+w*y)
    R[:, 1, 0] = 2*(x*y+w*z); R[:, 1, 1] = 1-2*(x*x+z*z); R[:, 1, 2] = 2*(y*z-w*x)
    R[:, 2, 0] = 2*(x*z-w*y); R[:, 2, 1] = 2*(y*z+w*x); R[:, 2, 2] = 1-2*(x*x+y*y)
    return R


def _R_quat(R):
    w = torch.sqrt(torch.clamp(1+R[0,0]+R[1,1]+R[2,2], min=1e-8))/2
    x = (R[2,1]-R[1,2])/(4*w); y = (R[0,2]-R[2,0])/(4*w); z = (R[1,0]-R[0,1])/(4*w)
    return torch.stack([w, x, y, z]).reshape(1, 4)


class _EEF64Helper:
    def __init__(self, base):
        self.b = base
        self.ikL = _build_ik(f"{A}/curobo_left.yml"); self.ikR = _build_ik(f"{A}/curobo_right.yml")
        self.tf = YamUnifiedEEFTransform(["default"])
        self._orig_ns = base._normalize_state
        self._orig_infer = base._infer_action_chunk
        base._normalize_state = self._normalize_state
        base._infer_action_chunk = self._infer_action_chunk

    def _fk_raw34(self, j14):
        def arm(ik, q6):
            st = ik.fk(torch.tensor(q6[None], device=_TDT.device, dtype=torch.float32))
            R = _quat_R(st.ee_quaternion)[0]; t = st.ee_position[0]
            M = torch.zeros(4, 4, device=R.device); M[:3, :3] = R; M[:3, 3] = t; M[3, 3] = 1
            return M.reshape(16)
        eL = arm(self.ikL, j14[0:6]); eR = arm(self.ikR, j14[7:13])
        return torch.cat([eL, eR, torch.tensor([j14[6]], device=eL.device),
                          torch.tensor([j14[13]], device=eL.device)]).float()

    def _normalize_state(self, v):
        v = np.asarray(v, dtype=np.float32)
        if v.shape[-1] == 14:
            v = self._fk_raw34(v).cpu().numpy()
        return self._orig_ns(v)

    def _ik_arm(self, ik, pos, R, seed6):
        goal = Pose(pos.to(_TDT.device), _R_quat(R).to(_TDT.device))
        seed = torch.tensor(seed6[None], device=_TDT.device, dtype=torch.float32)
        res = ik.solve_batch(goal, seed_config=seed.unsqueeze(1))
        return res.solution.squeeze(1)[0].cpu().numpy(), bool(res.success.view(-1)[0])

    def _infer_action_chunk(self, observation, instruction):
        j14 = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        act64 = np.asarray(self._orig_infer(observation, instruction), dtype=np.float32)
        base64 = self.tf._state64(self._fk_raw34(j14)).cpu()
        batch = {"state": {"default": base64[None, None]},
                 "action": {"default": torch.tensor(act64)[None]}}
        absa = self.tf.backward(batch)["action"]["default"][0]
        T = absa.shape[0]; out = np.zeros((T, 14), dtype=np.float32)
        seedL = j14[0:6].copy(); seedR = j14[7:13].copy()
        for t in range(T):
            dL = absa[t, 0:9]; dR = absa[t, 32:41]
            qL, okL = self._ik_arm(self.ikL, dL[0:3], _R_from_9d(dL), seedL)
            qR, okR = self._ik_arm(self.ikR, dR[0:3], _R_from_9d(dR), seedR)
            # Fall back to the last valid qpos when IK fails or returns non-finite — otherwise a
            # NaN/garbage qpos reaches SAPIEN's scene.step() and hard-segfaults the whole eval.
            if okL and np.isfinite(qL).all(): seedL = qL
            else: qL = seedL
            if okR and np.isfinite(qR).all(): seedR = qR
            else: qR = seedR
            out[t, 0:6] = qL; out[t, 6] = float(absa[t, 31]); out[t, 7:13] = qR; out[t, 13] = float(absa[t, 63])
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)   # belt-and-suspenders: no NaN to the sim


def install(base):
    _EEF64Helper(base)
    return base


def get_model(usr_args):
    base = base_dp.get_model(usr_args)
    if str(usr_args.get("action_space", os.environ.get("AGENTICWAM_ACTION_SPACE", ""))).lower() == "eef64":
        install(base)
    return base


def reset_model(policy):
    if hasattr(base_dp, "reset_model"):
        base_dp.reset_model(policy)
