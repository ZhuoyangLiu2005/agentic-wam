"""Convert the T-Rex gateway LeRobot v3.0 dataset (58-D joints) to an AgenticWAM LeRobot v2.1 shard.

Implements step 1 of docs/DATA.md for the Tianji + Sharpa layout (76-D raw):
    [0:16] L EEF pose 4x4 row-major | [16:32] R EEF pose | [32:54] L hand 22 | [54:76] R hand 22
Arm poses are FK of the 7 arm joints (state -> FK(state), action -> FK(action)), computed with T-Rex's VegaFK so the
frames match the T-Rex teleop recorder. Hand joints are passed through unchanged.

Videos: head_left -> cam_high (crop rows 0:280, cols 140:540, then resize to 224x320), left/right wrist -> full frame
resized to 224x320. h264 yuv420p 30 fps (libx264 defaults, as in the DATA.md ffmpeg example), one mp4 per episode.

Run (needs pin + av + pyarrow + pandas + opencv, e.g. T-Rex's uv project):
    uv run --project <T-Rex>/utils/fk_conversion python utils/convert_trex_v30_to_v21.py \
        --src data_local/trex_gateway_tong_transfer_sf_norawtac_20260820 --dst <out_dataset> --episodes 5
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_TREX = Path(__file__).resolve().parents[3] / "T-Rex"
FPS = 30
OUT_H, OUT_W = 224, 320
HEAD_CROP = (0, 280, 140, 540)  # rows y0:y1, cols x0:x1 on the 360x640 head_left frame -> 280x400, 0.8x to 224x320
CAMERAS = {  # source key -> (output key, crop or None)
    "observation.images.head_left": ("observation.images.cam_high", HEAD_CROP),
    "observation.images.left_wrist": ("observation.images.cam_left_wrist", None),
    "observation.images.right_wrist": ("observation.images.cam_right_wrist", None),
}
STATE_DIM = 76
STATS_FRAME_STRIDE = 10  # image stats are estimated from every 10th frame


def pack76(joints58: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """[N,58] joints + [N,2,4,4] FK poses -> [N,76] raw Tianji+Sharpa vector."""
    n = len(joints58)
    return np.concatenate(
        [poses[:, 0].reshape(n, 16), poses[:, 1].reshape(n, 16), joints58[:, 7:29], joints58[:, 36:58]], axis=1
    ).astype(np.float32)


def numeric_stats(a: np.ndarray) -> dict:
    a = a.astype(np.float64)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(),
            "std": a.std(0).tolist(), "count": [len(a)]}


def convert_video(trex, root, ep, src_key, dst_path, crop):
    """Decode one episode from the packed v3 mp4, crop/resize, encode a 224x320 h264 mp4. Returns (frames, sampled)."""
    path, start = trex.video_location(root, ep, src_key, FPS)
    n = int(ep.length)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sampled, count = [], 0
    with av.open(str(dst_path), "w", format="mp4", options={"movflags": "+faststart"}) as out:
        stream = out.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = OUT_W, OUT_H, "yuv420p"
        for rgb in trex.decode_frames(path, start, n, FPS, "rgb24"):
            assert rgb.shape == (360, 640, 3), f"unexpected frame shape {rgb.shape} in {src_key}"
            if crop is not None:
                y0, y1, x0, x1 = crop
                rgb = rgb[y0:y1, x0:x1]
            rgb = cv2.resize(rgb, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
            if count % STATS_FRAME_STRIDE == 0:
                sampled.append(rgb)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                out.mux(packet)
            count += 1
        for packet in stream.encode():
            out.mux(packet)
    assert count == n, f"{src_key}: wrote {count} frames, expected {n}"
    return np.stack(sampled)


def image_stats(frames: np.ndarray, n: int) -> dict:
    x = frames.astype(np.float64) / 255.0  # [S,H,W,3] -> per-channel stats, LeRobot shape (3,1,1)
    shape = lambda v: np.asarray(v).reshape(3, 1, 1).tolist()
    return {"min": shape(x.min((0, 1, 2))), "max": shape(x.max((0, 1, 2))), "mean": shape(x.mean((0, 1, 2))),
            "std": shape(x.std((0, 1, 2))), "count": [n]}


_WORKER = None


def _init_worker(trex_repo, urdf):
    global _WORKER
    sys.path.insert(0, str(Path(trex_repo) / "utils"))
    import unpack_lerobot_to_raw_fk as trex
    _WORKER = (trex, trex.VegaFK(urdf or trex.DEFAULT_URDF))


def convert_episode(root, shard, ep, new_idx, frame_offset, instruction):
    """Write one episode's parquet + 3 mp4s; return its episodes.jsonl row, stats row and manifest row."""
    trex, fk = _WORKER
    n = int(ep.length)
    arr = trex.episode_arrays(root, ep)
    state = pack76(arr["observation.state"], fk.poses(arr["observation.state"]))
    action = pack76(arr["action"], fk.poses(arr["action"]))
    assert state.shape == action.shape == (n, STATE_DIM) and np.isfinite(state).all() and np.isfinite(action).all()

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
    for src_key, (dst_key, crop) in CAMERAS.items():
        mp4 = shard / "videos/chunk-000" / dst_key / f"episode_{new_idx:06d}.mp4"
        stats[dst_key] = image_stats(convert_video(trex, root, ep, src_key, mp4, crop), n)
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):  # every non-string feature
        stats[key] = numeric_stats(cols[key].to_numpy()[:, None])
    return ({"episode_index": new_idx, "tasks": [instruction], "length": n},
            {"episode_index": new_idx, "stats": stats},
            {"episode_index": new_idx, "source_episode_index": int(ep.episode_index),
             "episode_uid": str(ep.episode_uid), "length": n})


def _convert_worker(job):
    root, shard, ep, new_idx, frame_offset, instruction = job
    return convert_episode(Path(root), Path(shard), pd.Series(ep), new_idx, frame_offset, instruction)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="v3.0 source dataset root")
    ap.add_argument("--dst", type=Path, required=True, help="output dataset root (train_shard_0 is created inside)")
    ap.add_argument("--episodes", default="all", help="'all', or N = first N episodes (pilot)")
    ap.add_argument("--trex_repo", type=Path, default=DEFAULT_TREX, help="T-Rex checkout (for VegaFK + video decode)")
    ap.add_argument("--urdf", type=Path, default=None, help="override T-Rex's default vega_1 URDF")
    ap.add_argument("--num_workers", type=int, default=1, help="episodes converted in parallel")
    args = ap.parse_args()

    root = args.src.resolve()
    shard = args.dst.resolve() / "train_shard_0"
    assert root not in shard.parents and root != shard, "output must be outside the source dataset"
    assert not (shard / "meta").exists(), f"{shard} already exists; choose a new --dst"

    info = json.loads((root / "meta/info.json").read_text())
    assert info["fps"] == FPS
    eps = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))])
    eps = eps.sort_values("episode_index")
    assert (eps.outcome == "success").all()
    if args.episodes != "all":
        eps = eps.iloc[: int(args.episodes)]
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    assert len(tasks) == 1, "converter assumes one dataset-wide instruction"
    instruction = str(tasks.index[0])

    (shard / "meta").mkdir(parents=True)
    lengths = eps.length.astype(int).to_numpy()
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    frame_offset = int(lengths.sum())
    jobs = [(str(root), str(shard), ep.to_dict(), i, int(offsets[i]), instruction)
            for i, (_, ep) in enumerate(eps.iterrows())]
    results = {}
    if args.num_workers == 1:
        _init_worker(args.trex_repo, args.urdf)
        for job in jobs:
            results[job[3]] = _convert_worker(job)
            print(f"episode {job[3]}: {job[2]['length']} frames ({len(results)}/{len(jobs)})", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker, initargs=(str(args.trex_repo), args.urdf)) as pool:
            futures = {pool.submit(_convert_worker, job): job[3] for job in jobs}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                print(f"episode {futures[future]}: done ({len(results)}/{len(jobs)})", flush=True)
    episodes_jsonl, stats_rows, manifest = (list(r) for r in zip(*(results[i] for i in range(len(jobs)))))

    video_feat = {"dtype": "video", "shape": [OUT_H, OUT_W, 3], "names": ["height", "width", "channels"],
                  "info": {"video.fps": FPS, "video.height": OUT_H, "video.width": OUT_W, "video.channels": 3,
                           "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                           "has_audio": False}}
    scalar = lambda dt: {"dtype": dt, "shape": [1], "names": None}
    features = {
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": None},
        "action": {"dtype": "float32", "shape": [STATE_DIM], "names": None},
        **{dst: video_feat for dst, _ in CAMERAS.values()},
        "timestamp": scalar("float32"), "frame_index": scalar("int64"), "episode_index": scalar("int64"),
        "index": scalar("int64"), "task_index": scalar("int64"),
    }
    meta = {
        "codebase_version": "v2.1", "robot_type": "tianji_sharpa", "fps": FPS, "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "total_episodes": len(eps), "total_frames": frame_offset, "total_tasks": 1,
        "total_videos": len(eps) * len(CAMERAS), "total_chunks": 1, "splits": {"train": f"0:{len(eps)}"},
        "features": features,
    }
    (shard / "meta/info.json").write_text(json.dumps(meta, indent=2) + "\n")
    (shard / "meta/episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in episodes_jsonl))
    (shard / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": instruction}) + "\n")
    (shard / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in stats_rows))
    (args.dst.resolve() / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"done: {len(eps)} episodes, {frame_offset} frames -> {shard}")


if __name__ == "__main__":
    main()
