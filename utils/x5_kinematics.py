"""ARX X5 arm kinematics (numpy): link6 pose in the arm's own base frame, its geometric Jacobian, and damped
least-squares IK.

The chain is RoboDojo's Assets/Robots/x5/X5A.urdf, joint1..joint6 -> link6. utils/convert_robodojo_v21_to_yam82.py
uses x5_fk to write the training poses (they match RoboDojo's recorded EE poses to ~0.1 mm), and
deploy/common/x5_eef64.py uses fk/ik to serve the post-trained checkpoint in RoboDojo, so both sides share one model.
"""
import numpy as np
from scipy.spatial.transform import Rotation

# (origin xyz, origin rpy, axis) per revolute joint, from X5A.urdf.
X5_CHAIN = (
    ((0, 0, 0.0605), (0, 0, 0), (0, 0, 1)),
    ((0.02, 0, 0.04), (0, 0, 0), (0, 1, 0)),
    ((-0.264, 0, 0), (3.1416, 0, 0), (0, 1, 0)),
    ((0.245, 0, -0.056), (0, 0, 0), (0, 1, 0)),
    ((0.06775, 0.0005, -0.0865), (0, 0, 0), (0, 0, 1)),
    ((0.02895, 0, 0.0865), (-3.1416, 0, 0), (1, 0, 0)),
)
_FIXED = [np.block([[Rotation.from_euler("xyz", rpy).as_matrix(), np.array(xyz)[:, None]], [np.zeros((1, 3)), 1]])
          for xyz, rpy, _ in X5_CHAIN]
_AXES = [np.asarray(ax, float) for _, _, ax in X5_CHAIN]


def x5_fk(q: np.ndarray) -> np.ndarray:
    """[N,6] arm joints -> [N,4,4] link6 pose in the arm base frame."""
    q = np.asarray(q, np.float64)
    n = len(q)
    m = np.tile(np.eye(4), (n, 1, 1))
    for fixed, ax, qi in zip(_FIXED, _AXES, q.T):
        j = np.tile(np.eye(4), (n, 1, 1))
        j[:, :3, :3] = Rotation.from_rotvec(ax[None] * qi[:, None]).as_matrix()
        m = m @ fixed @ j
    return m


def fk(q6) -> np.ndarray:
    """6 arm joints -> 4x4 link6 pose in the arm base frame."""
    return x5_fk(np.asarray(q6, np.float64)[None])[0]


def _fk_jac(q6):
    """(4x4 link6 pose, 6x6 geometric Jacobian [v; w] in the base frame)."""
    m = np.eye(4)
    z, p = [], []
    for fixed, ax, qi in zip(_FIXED, _AXES, q6):
        m = m @ fixed
        z.append(m[:3, :3] @ ax)
        p.append(m[:3, 3].copy())
        j = np.eye(4)
        j[:3, :3] = Rotation.from_rotvec(ax * qi).as_matrix()
        m = m @ j
    pe = m[:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        J[:3, i] = np.cross(z[i], pe - p[i])
        J[3:, i] = z[i]
    return m, J


def ik(T_target, init6, iters: int = 100, damp: float = 1e-4, pos_tol: float = 1e-3, rot_tol: float = 1e-2):
    """4x4 link6 target (base frame) -> (ok, 6 joints), damped least squares seeded at init6.

    ok means within pos_tol (m) and rot_tol (rad). The solver stops early at 0.1 mm / 1 mrad."""
    T_target = np.asarray(T_target, np.float64)
    Rt, tt = T_target[:3, :3], T_target[:3, 3]
    q = np.asarray(init6, np.float64).copy()
    ep = er = np.ones(3)
    for _ in range(iters):
        m, J = _fk_jac(q)
        ep = tt - m[:3, 3]
        er = Rotation.from_matrix(Rt @ m[:3, :3].T).as_rotvec()
        if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
            break
        dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), np.concatenate([ep, er]))
        q = q + np.clip(dq, -0.3, 0.3)
    return bool(np.linalg.norm(ep) < pos_tol and np.linalg.norm(er) < rot_tol), q


def rot6d_to_R(d6):
    a, b = d6[0:3], d6[3:6]
    c0 = a / np.linalg.norm(a)
    c1 = b - c0 * (c0 @ b)
    c1 = c1 / np.linalg.norm(c1)
    return np.stack([c0, c1, np.cross(c0, c1)], axis=1)


def eef_delta_9d(T_base, T_target):
    """Pose of T_target relative to T_base: [pos3 in the base pose's frame, first two columns of R_rel]."""
    Rb, tb = T_base[:3, :3], T_base[:3, 3]
    Rd = Rb.T @ T_target[:3, :3]
    return np.concatenate([Rb.T @ (T_target[:3, 3] - tb), Rd[:, 0], Rd[:, 1]])


def apply_delta_9d(T_base, d9):
    """Inverse of eef_delta_9d."""
    Rb, tb = T_base[:3, :3], T_base[:3, 3]
    T = np.eye(4)
    T[:3, :3] = Rb @ rot6d_to_R(np.asarray(d9[3:9], np.float64))
    T[:3, 3] = tb + Rb @ np.asarray(d9[0:3], np.float64)
    return T
