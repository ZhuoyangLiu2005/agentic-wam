"""YAM forward/inverse kinematics (REAL station model) + unified 64-D delta-EEF encode/decode.

station/station/station.xml is the authoritative bimanual model (both arms + real base mounts +
grasp/tcp sites + cameras). FK matches the recorded EEF EXACTLY (0.0 mm / 0.0 deg, absolute
world frame) with NO joint sign flip and EEF = *_grasp_site. mujoco-only (no mink/qpsolvers).

qpos layout (nq=16): left_joint1-6 [0:6], left fingers [6:8], right_joint1-6 [8:14], right fingers [14:16].
ABC 14-D: [Ljoint0-5, Lgrip(6), Rjoint0-5, Rgrip(13)] -> left=[0:6], right=[7:13].
unified 64-D/arm: [eef 9=pos3+rot6d | hand 22=0 | grip 1]; L=0:32 R=32:64; state=abs, action=delta.
"""
import os
import numpy as np
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(_HERE, "station", "station", "station.xml")
def _load_model_compat(path, max_fixes: int = 16):
    """Load the MJCF tolerating schema drift across mujoco versions.

    station.xml is authored for a newer mujoco than some envs ship (e.g. `texture colorspace`,
    `light type` are rejected by 3.3.x). Those attributes are purely visual and irrelevant to
    kinematics, so on a schema error we read the offending (element, attribute) straight out of
    the exception, strip it, and retry. The sanitized copy is written NEXT TO the original so the
    relative `meshdir` still resolves.
    """
    try:
        return mujoco.MjModel.from_xml_path(path)
    except ValueError as first:
        if "unrecognized attribute" not in str(first):
            raise
    import re
    src = open(path).read()
    alt = os.path.join(os.path.dirname(path), "_compat_" + os.path.basename(path))
    for _ in range(max_fixes):
        with open(alt, "w") as f:
            f.write(src)
        try:
            return mujoco.MjModel.from_xml_path(alt)
        except ValueError as e:
            m = re.search(r"unrecognized attribute: '([^']+)'.*?Element '([^']+)'", str(e), re.S)
            if not m:
                raise
            attr, elem = m.group(1), m.group(2)
            src = re.sub(r'(<%s\b[^>]*?)\s+%s="[^"]*"' % (re.escape(elem), re.escape(attr)), r"\1", src)
    raise RuntimeError("could not sanitize %s for mujoco %s" % (path, mujoco.__version__))


_m = _load_model_compat(XML)
_d = mujoco.MjData(_m)
def _sid(n): return mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_SITE, n)
# (qpos slice, dof indices, grasp-site id) per arm — dof order == qpos order for hinge joints
ARM = {"left":  (slice(0, 6),  list(range(0, 6)),  _sid("left_grasp_site")),
       "right": (slice(8, 14), list(range(8, 14)), _sid("right_grasp_site"))}


def _mat(sid):
    M = np.eye(4); M[:3, :3] = _d.site_xmat[sid].reshape(3, 3); M[:3, 3] = _d.site_xpos[sid]; return M

def fk(joints6, arm="left"):
    """arm 6-joint -> 4x4 EEF pose (grasp_site) in WORLD frame."""
    q, _, sid = ARM[arm]
    _d.qpos[:] = 0; _d.qpos[q] = np.asarray(joints6, float); mujoco.mj_kinematics(_m, _d)
    return _mat(sid)

def fk_both(qL6, qR6):
    _d.qpos[:] = 0; _d.qpos[0:6] = qL6; _d.qpos[8:14] = qR6; mujoco.mj_kinematics(_m, _d)
    return _mat(ARM["left"][2]), _mat(ARM["right"][2])

def ik(T_target, arm="left", init6=None, iters=100, damp=1e-4):
    """4x4 EEF target -> arm 6-joint via mujoco damped-least-squares. Returns (ok, joints6)."""
    q, dof, sid = ARM[arm]
    _d.qpos[:] = 0
    if init6 is not None: _d.qpos[q] = np.asarray(init6, float)
    Rt, tt = np.asarray(T_target)[:3, :3], np.asarray(T_target)[:3, 3]
    jacp, jacr = np.zeros((3, _m.nv)), np.zeros((3, _m.nv))
    ep = np.ones(3)
    for _ in range(iters):
        mujoco.mj_kinematics(_m, _d); mujoco.mj_comPos(_m, _d)
        pos, R = _d.site_xpos[sid].copy(), _d.site_xmat[sid].reshape(3, 3).copy()
        ep = tt - pos
        Re = Rt @ R.T
        er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
        if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
            break
        mujoco.mj_jacSite(_m, _d, jacp, jacr, sid)
        J = np.vstack([jacp[:, dof], jacr[:, dof]])
        dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), np.concatenate([ep, er]))
        _d.qpos[q] = _d.qpos[q] + np.clip(dq, -0.3, 0.3)
    return bool(np.linalg.norm(ep) < 1e-3), _d.qpos[q].copy()


def state14_to_82(s14):
    s14 = np.asarray(s14, float); o = np.zeros(82, np.float32)
    TL, TR = fk_both(s14[0:6], s14[7:13])
    o[0:16] = TL.reshape(16); o[16:32] = TR.reshape(16)
    o[32] = s14[6]; o[33] = s14[13]; o[34:40] = s14[0:6]; o[40:46] = s14[7:13]
    return o


def rt_to_6d(R): return np.concatenate([R[:, 0], R[:, 1]])
def rot6d_to_R(d6):
    a, b = d6[0:3], d6[3:6]
    c0 = a / np.linalg.norm(a); c1 = b - c0 * (c0 @ b); c1 = c1 / np.linalg.norm(c1)
    return np.stack([c0, c1, np.cross(c0, c1)], axis=1)
def eef_delta_9d(T_base, T_target):
    Rb, tb = T_base[:3, :3], T_base[:3, 3]; Ra, ta = T_target[:3, :3], T_target[:3, 3]
    Rd = Rb.T @ Ra
    return np.concatenate([Rb.T @ (ta - tb), Rd[:, 0], Rd[:, 1]])
def apply_delta_9d(T_base, d9):
    Rb, tb = T_base[:3, :3], T_base[:3, 3]
    Ra = Rb @ rot6d_to_R(d9[3:9]); ta = tb + Rb @ d9[0:3]
    T = np.eye(4); T[:3, :3] = Ra; T[:3, 3] = ta; return T


if __name__ == "__main__":
    for arm in ("left", "right"):
        q = np.array([0.2, -0.4, 0.3, -0.1, 0.5, 0.0])
        T = fk(q, arm); ok, q2 = ik(T, arm, init6=q + 0.15)
        print("[%s] FK/IK ok=%s max|dq|=%.6f" % (arm, ok, np.abs(q - q2).max()))
    Tb, Tt = fk(np.zeros(6), "left"), fk(np.ones(6) * 0.1, "left")
    d9 = eef_delta_9d(Tb, Tt)
    print("delta9 round-trip pos err = %.6f m" % np.linalg.norm(Tt[:3, 3] - apply_delta_9d(Tb, d9)[:3, 3]))
