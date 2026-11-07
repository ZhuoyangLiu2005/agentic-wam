"""FK/IK check of a RoboDojo task's training labels (ARX X5).

1. utils/x5_kinematics.X5_CHAIN vs an FK parsed independently from RoboDojo's Assets/Robots/x5/X5A.urdf (the URDF the
   sim's robot_config.yml names: base_link -> link6), on every training frame.
2. Shard labels: stored link6 poses vs FK(stored joints), for state and action; action[t] == state[t+1]; the shard
   joints/grippers vs the RoboDojo v2.1 source (14-D).
3. Training delta label (YamUnifiedEEFTransform.forward, chunk-start anchored) vs the inference encoder
   (x5_eef64.joints_to_delta64), and the inference decoder (decode_chunk: apply_delta_9d + IK) on the TRUE label:
   recovered joints and FK pose vs the recorded action.
"""
import glob
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rd_env import REPO, SHARD, SRC, TASK, TRAIN_EPS  # noqa: E402
sys.path[:0] = [str(REPO / "utils"), str(REPO / "deploy/common"), str(REPO / "src")]
import x5_kinematics as xk  # noqa: E402
import x5_eef64  # noqa: E402
from agenticwam.datasets.lerobot.transforms.yam_unified_eef import YamUnifiedEEFTransform  # noqa: E402

URDF = REPO / "RoboDojo/Assets/Robots/x5/X5A.urdf"
H, STRIDE = 32, 24


def urdf_chain(path, base="base_link", tip="link6"):
    root = ET.parse(path).getroot()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    chain, link = [], tip
    while link != base:
        j = by_child[link]
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()]
        ax = j.find("axis")
        axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [1, 0, 0]
        chain.append((j.get("name"), j.get("type"), xyz, rpy, axis))
        link = j.find("parent").get("link")
    return chain[::-1]


def urdf_fk(chain, q):
    out = []
    for qq in q:
        m, k = np.eye(4), 0
        for _, typ, xyz, rpy, axis in chain:
            f = np.eye(4); f[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix(); f[:3, 3] = xyz   # URDF rpy = fixed-axis XYZ
            m = m @ f
            if typ in ("revolute", "continuous"):
                r = np.eye(4); r[:3, :3] = Rotation.from_rotvec(np.asarray(axis) * qq[k]).as_matrix(); m = m @ r; k += 1
        out.append(m)
    return np.array(out)


def rot_err_deg(A, B):
    R = np.einsum("nji,njk->nik", A[:, :3, :3], B[:, :3, :3])
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))


def j14(s82):
    return np.concatenate([s82[:, 34:40], s82[:, 32:33], s82[:, 40:46], s82[:, 33:34]], 1).astype(np.float64)


def main():
    rep = {}
    chain = urdf_chain(URDF)
    rep["urdf_chain"] = [c[:2] for c in chain]
    trained = set(json.load(open(TRAIN_EPS)))
    eps = {}
    for p in sorted(glob.glob(str(SHARD / "data/chunk-000/episode_*.parquet"))):
        e = int(Path(p).stem.split("_")[1])
        if e in trained:
            d = pd.read_parquet(p)
            eps[e] = (np.stack(d["observation.state"]).astype(np.float64), np.stack(d["action"]).astype(np.float64))
    S = np.concatenate([v[0] for v in eps.values()]); A = np.concatenate([v[1] for v in eps.values()])

    # 1. code chain vs URDF
    out = {}
    for side, js in (("L", slice(34, 40)), ("R", slice(40, 46))):
        a, b = xk.x5_fk(S[:, js]), urdf_fk(chain, S[:, js])
        out[side] = {"pos_mm_max": float(np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=1).max() * 1e3),
                     "rot_deg_max": float(rot_err_deg(a, b).max())}
    rep["1_x5_kinematics_vs_X5A_urdf"] = out

    # 2. stored labels
    out = {}
    for name, X in (("state", S), ("action", A)):
        for side, ps, js in (("L", slice(0, 16), slice(34, 40)), ("R", slice(16, 32), slice(40, 46))):
            st = X[:, ps].reshape(-1, 4, 4); fk = xk.x5_fk(X[:, js])
            out[f"{name}_{side}"] = {"pos_mm_max": float(np.linalg.norm(st[:, :3, 3] - fk[:, :3, 3], axis=1).max() * 1e3),
                                     "rot_deg_max": float(rot_err_deg(st, fk).max())}
    nxt = max(np.abs(a[:-1, 32:46] - s[1:, 32:46]).max() for s, a in eps.values())
    out["action_t_vs_state_t+1_maxabs"] = float(nxt)
    rep["2_stored_labels"] = out

    # 3. training delta vs inference encode/decode, every replan window
    tf = YamUnifiedEEFTransform(keys=["default"])
    enc_err, pos_err, rot_err, q_err, fails, n_rows = [], [], [], [], 0, 0
    for e, (s, a) in eps.items():
        for t in range(0, len(s) - 1, STRIDE):
            k = min(H, len(s) - t)
            st = torch.as_tensor(s[t:t + 1][None]); ac = torch.as_tensor(a[t:t + k][None])
            b = tf.forward({"state": {"default": st}, "action": {"default": ac}})
            lab = b["action"]["default"][0].numpy()                          # [k,64] training label
            enc = x5_eef64.joints_to_delta64(j14(a[t:t + k]), j14(s[t:t + 1])[0])
            enc_err.append(np.abs(lab - enc).max())
            q, f = x5_eef64.decode_chunk(lab, j14(s[t:t + 1])[0])
            fails += f; n_rows += 2 * k
            gt = j14(a[t:t + k])
            for js in (slice(0, 6), slice(7, 13)):
                P, G = xk.x5_fk(q[:, js]), xk.x5_fk(gt[:, js])
                pos_err.append(np.linalg.norm(P[:, :3, 3] - G[:, :3, 3], axis=1)); rot_err.append(rot_err_deg(P, G))
                q_err.append(np.abs(q[:, js] - gt[:, js]).max(1))
            q_err.append(np.abs(q[:, [6, 13]] - gt[:, [6, 13]]).max(1))
    pos_err, rot_err, q_err = map(np.concatenate, (pos_err, rot_err, q_err))
    rep["3_label_vs_inference"] = {
        "train_label_vs_joints_to_delta64_maxabs": float(max(enc_err)),
        "decode_chunk_on_true_label": {"rows": int(n_rows), "ik_failures": int(fails),
                                        "pos_mm_mean": float(pos_err.mean() * 1e3), "pos_mm_max": float(pos_err.max() * 1e3),
                                        "rot_deg_mean": float(rot_err.mean()), "rot_deg_max": float(rot_err.max()),
                                        "joint_rad_maxabs_p99": float(np.percentile(q_err, 99)),
                                        "joint_rad_maxabs_max": float(q_err.max())}}
    rep["n_episodes"], rep["n_frames"] = len(eps), int(len(S))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
