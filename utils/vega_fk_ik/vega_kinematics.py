"""Vega (Tianji arms + Sharpa hands) forward/inverse kinematics.

Vendored and adapted from the T-Rex repo so the deploy server has no runtime dependency on a
T-Rex checkout: FK is `VegaFK` (T-Rex/utils/unpack_lerobot_to_raw_fk.py), IK is `PinkLocalIK` +
`build_reduced_bimanual_robot` (T-Rex/hardware_code/teleop/ik_utils.py +
.../teleop/robot_descriptions.py). Only the 7-DOF-per-arm kinematics are vendored -- hand joints
need no FK/IK (see deploy/common/vega_eef64.py), so no hand URDF/geometry is needed here.

Needs `pin>=3.2` (pinocchio), `pin-pink`, `qpsolvers` and `daqp` (the solver="daqp" backend) installed in
the deploy venv, as in T-Rex hardware_code/requirements.txt -- same ad hoc-dependency situation as
`mujoco` for utils/yam_fk_ik/yam_kinematics.py. Neither `coal`/`fcl` nor
`dexmate-urdf` are required: IK here never builds a GeometryModel (collision avoidance is
commented out in the upstream `PinkLocalIK.solve_ik` too), so only the kinematic tree -- the
~30KB `urdf/vega_1.urdf` text file vendored alongside this module (Apache-2.0, see
urdf/LICENSE.dexmate-urdf) -- is ever loaded, never mesh geometry.

API mirrors utils/yam_fk_ik/yam_kinematics.py's shape, except IK solves both arms in one call
because the vendored solver (Pink) always does -- there is no independent per-arm IK upstream:
    fk_both(qL7, qR7) -> (T_L, T_R)                              4x4 EEF poses, WORLD frame
    ik_both(T_L, T_R, initL7, initR7) -> (ok, qL7, qR7)
"""
from pathlib import Path

import numpy as np
import pinocchio as pin

_HERE = Path(__file__).resolve().parent
URDF = _HERE / "urdf" / "vega_1.urdf"

TORSO = (0.9, 1.57, 0.1)   # fixed torso pose used throughout data conversion (CHOICES.md)
HEAD = (0.28, 0.0, 0.0)
# IK posture-regularization target = the teleop home pose (T-Rex hardware_code/config/default.yaml),
# not zeros: the arms are 7-DOF redundant, so this target picks the null-space solution.
LEFT_ARM_HOME = (0.84, 0.51, 0.37, -1.30, -0.65, -0.29, -0.03)
RIGHT_ARM_HOME = (-0.84, -0.51, -0.37, -1.30, 0.65, 0.29, 0.03)
ARM_JOINT_SUFFIXES = [f"arm_j{i}" for i in range(1, 8)]   # arm_j1..arm_j7 (7 DOF; j8 unused, as upstream)
EE_FRAMES = ("L_ee", "R_ee")


def _joint_index(model, name):
    assert model.existJointName(name), f"missing joint {name!r} in {URDF.name}"
    joint = model.joints[model.getJointId(name)]
    assert joint.nq == 1, f"expected scalar joint {name!r}"
    return joint.idx_q


class _FK:
    """Named-joint FK using the same wrist frames as the teleop recorder (VegaFK)."""

    def __init__(self, urdf=URDF, torso=TORSO, head=HEAD):
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        for prefix, values in (("torso", torso), ("head", head)):
            for i, v in enumerate(values, 1):
                self.q[_joint_index(self.model, f"{prefix}_j{i}")] = v
        self.arm_indices = [
            [_joint_index(self.model, f"{side}_{suffix}") for suffix in ARM_JOINT_SUFFIXES]
            for side in ("L", "R")
        ]
        self.frame_ids = []
        for frame in EE_FRAMES:
            assert self.model.existFrame(frame), f"missing FK frame {frame!r}"
            self.frame_ids.append(self.model.getFrameId(frame))

    def both(self, qL7, qR7):
        self.q[self.arm_indices[0]] = np.asarray(qL7, dtype=np.float64)
        self.q[self.arm_indices[1]] = np.asarray(qR7, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        return tuple(self.data.oMf[fid].homogeneous.copy() for fid in self.frame_ids)


class _ReducedArms:
    """Arms-only model (torso/head locked at TORSO/HEAD) for IK -- build_reduced_bimanual_robot,
    kinematics-only: no GeometryModel is built (matches upstream solve_ik, which never uses one)."""

    def __init__(self, urdf=URDF, torso=TORSO, head=HEAD):
        full = pin.buildModelFromUrdf(str(urdf))
        head_idx = np.array([_joint_index(full, f"head_j{i}") for i in range(1, 4)])
        torso_idx = np.array([_joint_index(full, f"torso_j{i}") for i in range(1, 4)])
        default_qpos = pin.neutral(full)
        default_qpos[head_idx] = head
        default_qpos[torso_idx] = torso

        arm_joint_names = {f"{side}_{suffix}" for side in ("L", "R") for suffix in ARM_JOINT_SUFFIXES}
        lock_ids = [full.getJointId(n) for n in full.names[1:] if n not in arm_joint_names]
        self.model = pin.buildReducedModel(full, lock_ids, default_qpos)
        self.data = self.model.createData()

        self.left_idx = np.array([_joint_index(self.model, f"L_{s}") for s in ARM_JOINT_SUFFIXES])
        self.right_idx = np.array([_joint_index(self.model, f"R_{s}") for s in ARM_JOINT_SUFFIXES])
        assert self.model.nq == self.model.nv == 2 * len(ARM_JOINT_SUFFIXES)
        self.default_qpos = self.assemble(LEFT_ARM_HOME, RIGHT_ARM_HOME)

    def assemble(self, qL7, qR7):
        qpos = np.zeros(self.model.nq)
        qpos[self.left_idx] = qL7
        qpos[self.right_idx] = qR7
        return qpos

    def disassemble(self, qpos):
        return qpos[self.left_idx].copy(), qpos[self.right_idx].copy()


_fk = _FK()
_arms = _ReducedArms()


def fk_both(qL7, qR7):
    """Both arms' 7 joint angles -> (T_L, T_R) 4x4 EEF poses (L_ee/R_ee frames), WORLD frame."""
    return _fk.both(qL7, qR7)


def ik_both(T_L_target, T_R_target, initL7, initR7, iters: int = 5, dt: float = 0.05):
    """Both arms' 4x4 EEF targets -> (ok, qL7, qR7) via Pink differential IK (PinkLocalIK),
    seeded at (initL7, initR7). Solves jointly (one 14-D configuration) -- the vendored solver
    has no independent per-arm mode. `ok` is a 1mm position-tolerance check (yam_kinematics.ik's
    convention); the upstream teleop solver has no such check since any incremental result is
    usable there."""
    from pink import Configuration
    from pink import solve_ik as _pink_solve_ik
    from pink.tasks import FrameTask, PostureTask

    initial_qpos = _arms.assemble(initL7, initR7)
    left_task = FrameTask("L_ee", position_cost=50.0, orientation_cost=1.0, lm_damping=0, gain=0.2)
    right_task = FrameTask("R_ee", position_cost=50.0, orientation_cost=1.0, lm_damping=0, gain=0.2)
    smoothness = PostureTask(cost=0.2, lm_damping=0.0, gain=0.2)
    regularization = PostureTask(cost=0.05, lm_damping=0.0, gain=0.2)
    smoothness.set_target(initial_qpos)
    regularization.set_target(_arms.default_qpos)
    left_task.set_target(pin.SE3(np.asarray(T_L_target, dtype=np.float64)))
    right_task.set_target(pin.SE3(np.asarray(T_R_target, dtype=np.float64)))
    tasks = [left_task, right_task, smoothness, regularization]

    configuration = Configuration(model=_arms.model, data=_arms.data, q=initial_qpos,
                                  copy_data=True, forward_kinematics=True)
    for _ in range(iters):
        try:
            velocity = _pink_solve_ik(configuration=configuration, tasks=tasks, dt=dt,
                                      solver="daqp", damping=0)
            if np.any(np.isnan(velocity)):
                velocity = np.zeros_like(configuration.q)
        except Exception:
            velocity = np.zeros_like(configuration.q)
        configuration.integrate_inplace(velocity, dt)
        clipped = np.clip(configuration.q, _arms.model.lowerPositionLimit, _arms.model.upperPositionLimit)
        configuration.update(clipped)

    qL7, qR7 = _arms.disassemble(configuration.q)
    T_L_actual, T_R_actual = fk_both(qL7, qR7)
    pos_err = max(np.linalg.norm(T_L_actual[:3, 3] - np.asarray(T_L_target)[:3, 3]),
                  np.linalg.norm(T_R_actual[:3, 3] - np.asarray(T_R_target)[:3, 3]))
    return bool(pos_err < 1e-3), qL7, qR7


if __name__ == "__main__":
    qL = np.array([0.2, -0.4, 0.3, -0.1, 0.5, 0.0, 0.1])
    qR = np.array([-0.2, 0.4, -0.3, 0.1, -0.5, 0.0, -0.1])
    T_L, T_R = fk_both(qL, qR)
    ok, qL2, qR2 = ik_both(T_L, T_R, qL + 0.1, qR + 0.1)
    print(f"FK/IK ok={ok} max|dqL|={np.abs(qL - qL2).max():.6f} max|dqR|={np.abs(qR - qR2).max():.6f}")
