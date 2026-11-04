"""Convert the RoboDojo sim LeRobot v2.1 release (dual ARX X5, 14-D joints) to AgenticWAM SFT shards in the YAM 82-D layout.

Implements step 1 of docs/DATA.md. The source (`bash scripts/RoboDojo/download_data.sh huggingface lerobot_v2.1`) is one
v2.1 dataset: 35 tasks x 100 episodes, 25 Hz, state/action = [L joint0..5, L gripper, R joint0..5, R gripper] with
action[t] = state[t+1] (absolute), AV1 480x640 videos keyed cam_high / cam_left_wrist / cam_right_wrist.

Output raw 82-D (the YAM layout, so `yam_unified_eef.YamUnifiedEEFTransform`, configs/real_post_train/data/yam.yaml and
scripts/train_real_yam.sh apply unchanged):
    [0:16]  L link6 pose 4x4 row-major, in the LEFT arm's base frame   (FK of the 6 arm joints, X5A.urdf)
    [16:32] R link6 pose 4x4 row-major, in the RIGHT arm's base frame
    [32] L gripper, [33] R gripper   (RoboDojo ee joint state: 1 = open, ~0.08 = closed; YAM: 0 closed .. 1 open)
    [34:40] L arm joints, [40:46] R arm joints   (kept for IK / reference; unused by the transform)
    [46:82] zero
The FK matches RoboDojo's recorded state/*_ee_poses (world frame = spawn pose @ FK) to ~0.1 mm on the demo HDF5s.

Videos: crop rows 16:464 (480x640 -> 448x640, the 224:320 aspect), resize to 224x320 (area), h264 yuv420p, 25 fps.
fps stays 25: the loader indexes frames by each shard's fps and the RoboDojo sim steps at 25 Hz.
Shards: one per task (train_shard_<task_index>), episodes renumbered from 0 inside each shard.

    ~/venvs/agenticwam-data/bin/python utils/convert_robodojo_v21_to_yam82.py \
        --src RoboDojo/data/RoboDojo_lerobot_v21_video --dst data_local/robodojo_x5_yam82 --episodes_per_task 2 --tasks 0,1
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x5_kinematics import x5_fk  # noqa: E402  (X5A.urdf joint1..6 -> link6, shared with deploy/common/x5_eef64.py)

FPS = 25
SRC_H, SRC_W = 480, 640
OUT_H, OUT_W = 224, 320
CROP_Y0, CROP_H = 16, 448
CAMERAS = ("observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist")
RAW_DIM = 82
STATS_FRAME_STRIDE = 10


def pack82(j14: np.ndarray) -> np.ndarray:
    """[N,14] RoboDojo joints -> [N,82] raw YAM-layout vector."""
    n = len(j14)
    qL, gL, qR, gR = j14[:, 0:6], j14[:, 6], j14[:, 7:13], j14[:, 13]
    out = np.zeros((n, RAW_DIM), np.float64)
    out[:, 0:16] = x5_fk(qL).reshape(n, 16)
    out[:, 16:32] = x5_fk(qR).reshape(n, 16)
    out[:, 32], out[:, 33] = gL, gR
    out[:, 34:40], out[:, 40:46] = qL, qR
    return out.astype(np.float32)


def numeric_stats(a: np.ndarray) -> dict:
    a = a.astype(np.float64)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(),
            "std": a.std(0).tolist(), "count": [len(a)]}


def convert_video(src: Path, dst: Path, n: int) -> dict:
    """AV1 480x640 -> cropped 224x320 h264; returns LeRobot image stats over every STATS_FRAME_STRIDE-th frame."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"crop={SRC_W}:{CROP_H}:0:{CROP_Y0},scale={OUT_W}:{OUT_H}:flags=area"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-threads", "2", "-i", str(src), "-vf", vf,
                    "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
                   check=True)
    cap, frames, count = cv2.VideoCapture(str(dst)), [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if count % STATS_FRAME_STRIDE == 0:
            frames.append(bgr[..., ::-1])
        count += 1
    cap.release()
    assert count == n, f"{dst}: {count} frames, parquet has {n}"
    x = np.stack(frames).astype(np.float64) / 255.0
    shape = lambda v: np.asarray(v).reshape(3, 1, 1).tolist()
    return {"min": shape(x.min((0, 1, 2))), "max": shape(x.max((0, 1, 2))), "mean": shape(x.mean((0, 1, 2))),
            "std": shape(x.std((0, 1, 2))), "count": [n]}


def convert_episode(src_root: str, chunks_size: int, shard: str, src_ep: int, new_idx: int, frame_offset: int):
    src_root, shard = Path(src_root), Path(shard)
    chunk = f"chunk-{src_ep // chunks_size:03d}"
    df = pd.read_parquet(src_root / "data" / chunk / f"episode_{src_ep:06d}.parquet")
    n = len(df)
    state = pack82(np.stack(df["observation.state"].to_numpy()).astype(np.float64))
    action = pack82(np.stack(df["action"].to_numpy()).astype(np.float64))
    assert np.isfinite(state).all() and np.isfinite(action).all()
    cols = {
        "observation.state": pa.array(list(state), pa.list_(pa.float32())),
        "action": pa.array(list(action), pa.list_(pa.float32())),
        "timestamp": pa.array(np.arange(n, dtype=np.float32) / FPS),
        "frame_index": pa.array(np.arange(n, dtype=np.int64)),
        "episode_index": pa.array(np.full(n, new_idx, dtype=np.int64)),
        "index": pa.array(np.arange(frame_offset, frame_offset + n, dtype=np.int64)),
        "task_index": pa.array(np.zeros(n, dtype=np.int64)),
    }
    out = shard / "data/chunk-000" / f"episode_{new_idx:06d}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), out)
    stats = {"observation.state": numeric_stats(state), "action": numeric_stats(action)}
    for key in CAMERAS:
        stats[key] = convert_video(src_root / "videos" / chunk / key / f"episode_{src_ep:06d}.mp4",
                                   shard / "videos/chunk-000" / key / f"episode_{new_idx:06d}.mp4", n)
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        stats[key] = numeric_stats(cols[key].to_numpy()[:, None])
    return new_idx, n, {"episode_index": new_idx, "stats": stats}, int(df["task_index"].iloc[0])


def write_meta(shard: Path, instruction: str, lengths: list[int], stats_rows: list[dict]):
    video_feat = {"dtype": "video", "shape": [OUT_H, OUT_W, 3], "names": ["height", "width", "channels"],
                  "info": {"video.fps": FPS, "video.height": OUT_H, "video.width": OUT_W, "video.channels": 3,
                           "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                           "has_audio": False}}
    scalar = lambda dt: {"dtype": dt, "shape": [1], "names": None}
    features = {
        "observation.state": {"dtype": "float32", "shape": [RAW_DIM], "names": None},
        "action": {"dtype": "float32", "shape": [RAW_DIM], "names": None},
        **{k: video_feat for k in CAMERAS},
        "timestamp": scalar("float32"), "frame_index": scalar("int64"), "episode_index": scalar("int64"),
        "index": scalar("int64"), "task_index": scalar("int64"),
    }
    n_ep = len(lengths)
    meta = {
        "codebase_version": "v2.1", "robot_type": "dual_arx_x5", "fps": FPS, "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "total_episodes": n_ep, "total_frames": int(sum(lengths)), "total_tasks": 1,
        "total_videos": n_ep * len(CAMERAS), "total_chunks": 1, "splits": {"train": f"0:{n_ep}"},
        "features": features,
    }
    (shard / "meta").mkdir(parents=True, exist_ok=True)
    (shard / "meta/info.json").write_text(json.dumps(meta, indent=2) + "\n")
    (shard / "meta/episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": i, "tasks": [instruction], "length": L}) + "\n"
                for i, L in enumerate(lengths)))
    (shard / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": instruction}) + "\n")
    (shard / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in stats_rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="RoboDojo_lerobot_v21_video root")
    ap.add_argument("--dst", type=Path, required=True, help="output dataset root (train_shard_<task> inside)")
    ap.add_argument("--tasks", default="all", help="'all' or comma-separated source task indices")
    ap.add_argument("--episodes_per_task", type=int, default=0, help="0 = all; N = first N per task (pilot)")
    ap.add_argument("--num_workers", type=int, default=16)
    args = ap.parse_args()

    src = args.src.resolve()
    info = json.loads((src / "meta/info.json").read_text())
    assert info["fps"] == FPS and info["features"]["observation.state"]["shape"] == [14], info["features"]["observation.state"]
    tasks = {r["task_index"]: r["task"] for r in map(json.loads, (src / "meta/tasks.jsonl").read_text().splitlines())}
    task_of = {t: i for i, t in tasks.items()}
    by_task = defaultdict(list)
    for r in map(json.loads, (src / "meta/episodes.jsonl").read_text().splitlines()):
        assert len(r["tasks"]) == 1, r
        by_task[task_of[r["tasks"][0]]].append((r["episode_index"], r["length"]))
    want = sorted(by_task) if args.tasks == "all" else [int(t) for t in args.tasks.split(",")]

    jobs = []  # (task, src_ep, new_idx, frame_offset, length)
    for t in want:
        eps = sorted(by_task[t])[: args.episodes_per_task or None]
        shard = args.dst.resolve() / f"train_shard_{t}"
        assert not (shard / "meta").exists(), f"{shard} already exists; choose a new --dst"
        off = 0
        for i, (e, L) in enumerate(eps):
            jobs.append((t, e, i, off, L)); off += L

    results = defaultdict(dict)
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futs = {pool.submit(convert_episode, str(src), info["chunks_size"],
                            str(args.dst.resolve() / f"train_shard_{t}"), e, i, off): (t, L) for t, e, i, off, L in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            t, L = futs[fut]
            new_idx, n, stats_row, src_task = fut.result()
            assert n == L and src_task == t, (t, new_idx, n, L, src_task)
            results[t][new_idx] = (n, stats_row)
            if k % 50 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} episodes", flush=True)

    manifest = []
    for t in want:
        rows = [results[t][i] for i in range(len(results[t]))]
        write_meta(args.dst.resolve() / f"train_shard_{t}", tasks[t], [r[0] for r in rows], [r[1] for r in rows])
        manifest += [{"shard": f"train_shard_{t}", "episode_index": i, "source_episode_index": e}
                     for tt, e, i, _, _ in jobs if tt == t]
    (args.dst.resolve() / "source_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"done: {len(want)} shards, {len(jobs)} episodes -> {args.dst}")


if __name__ == "__main__":
    main()
