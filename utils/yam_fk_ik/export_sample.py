"""Export one episode in raw-joint + 64-D EEF form for real-robot IK replay & validation.
Uses the REAL station model (yam_kinematics) — absolute EEF matches the recording (world frame), no sign flip."""
import os, sys, json, shutil
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yam_kinematics as K
ABC = os.environ.get("YAM_JOINT_DATASET", "/path/to/yam_joint_lerobot")   # a 14-D joint YAM LeRobot dataset
SHARD, EP = "train_shard_0", 0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_trajectory")
os.makedirs(OUT, exist_ok=True)
df = pd.read_parquet(f"{ABC}/{SHARD}/data/chunk-000/episode_{EP:06d}.parquet")
st = np.stack([np.asarray(x, np.float32) for x in df["observation.state"].values])
ac = np.stack([np.asarray(x, np.float32) for x in df["action"].values])
T = len(st); s64 = np.zeros((T, 64), np.float32); a64 = np.zeros((T, 64), np.float32); ikj = np.zeros((T, 14), np.float32)
errs = []
for t in range(T):
    TLs, TRs = K.fk(st[t, 0:6], "left"), K.fk(st[t, 7:13], "right")
    TLa, TRa = K.fk(ac[t, 0:6], "left"), K.fk(ac[t, 7:13], "right")
    s64[t, 0:3] = TLs[:3, 3]; s64[t, 3:9] = K.rt_to_6d(TLs[:3, :3]); s64[t, 31] = st[t, 6]
    s64[t, 32:35] = TRs[:3, 3]; s64[t, 35:41] = K.rt_to_6d(TRs[:3, :3]); s64[t, 63] = st[t, 13]
    dL, dR = K.eef_delta_9d(TLs, TLa), K.eef_delta_9d(TRs, TRa)
    a64[t, 0:9] = dL; a64[t, 31] = ac[t, 6]; a64[t, 32:41] = dR; a64[t, 63] = ac[t, 13]
    _, qL = K.ik(K.apply_delta_9d(TLs, dL), "left", init6=st[t, 0:6])
    _, qR = K.ik(K.apply_delta_9d(TRs, dR), "right", init6=st[t, 7:13])
    ikj[t, 0:6] = qL; ikj[t, 6] = ac[t, 6]; ikj[t, 7:13] = qR; ikj[t, 13] = ac[t, 13]
    errs.append(float(np.abs(np.r_[qL, qR] - np.r_[ac[t, 0:6], ac[t, 7:13]]).max()))
for n, arr in [("raw_state_14d", st), ("raw_action_14d", ac), ("state_64d", s64),
               ("action_delta_64d", a64), ("ik_replay_joints_14d", ikj)]:
    np.save(os.path.join(OUT, n + ".npy"), arr)
for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
    src = f"{ABC}/{SHARD}/videos/chunk-000/observation.images.{cam}/episode_{EP:06d}.mp4"
    if os.path.exists(src): shutil.copy(src, os.path.join(OUT, cam + ".mp4"))
json.dump({"episode": f"{SHARD}/episode_{EP:06d}", "frames": int(T), "fps": 30,
           "model": "station/station/station.xml (REAL bimanual URDF; abs EEF matches the recording's world frame)",
           "eef_site": "left_grasp_site / right_grasp_site", "sign_flip": "none",
           "layout_14d": "[Lj0-5, Lgrip(6), Rj0-5, Rgrip(13)]",
           "layout_64d": "per-arm [eef 9=pos3+rot6d | hand 22=0 | grip 1]; L=0:32 R=32:64; state=abs, action=delta-in-current-frame",
           "ik_roundtrip_maxjoint_err_rad_mean": float(np.mean(errs)),
           "ik_roundtrip_maxjoint_err_rad_p95": float(np.percentile(errs, 95))},
          open(os.path.join(OUT, "meta.json"), "w"), indent=2)
print("EXPORT_DONE frames=%d ik_err_mean=%.5f rad p95=%.5f rad" % (T, np.mean(errs), np.percentile(errs, 95)))
