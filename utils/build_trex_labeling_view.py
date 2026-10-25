"""Build the Route A LABELING view for a dataset made by convert_trex_v30_to_v21.py (docs/CODE_ANNOTATION.md §4.2).

The `tianji_sharpa` engine embodiment (codegen/pretrain/pipeline/codegen/data/tianji_sharpa_reader.py) reads one flat
LeRobot dir with:
    observation.state [78] = [0:9] L wrist EEF | [9:18] R wrist EEF | [18:78] per-finger tactile f6 (L then R,
                             thumb, index, middle, ring, pinky; 6 axes each), named in meta/modality.json
    videos/chunk-000/top/  = head_left at the native 360x640, no crop (the VLM reads it, the policy never does)
Episode i here is episode i of the training view (same order, same lengths), so labels map back frame for frame.

Wrist EEF = (t, R col0, R col1) of the training view's pose16 -- the exact FK the policy trains on, not a recompute.
Tactile comes from the source v3.0 dataset (the training view drops it), matched via source_manifest.json.

Run (needs av + pyarrow + pandas, e.g. T-Rex's uv project):
    python utils/build_trex_labeling_view.py --src <v3.0 source> --train_view <converted dataset> --dst <label view>
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_TREX = Path(__file__).resolve().parents[3] / "T-Rex"
FPS = 30
TOP_H, TOP_W = 360, 640
TOP_SRC = "observation.images.head_left"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
EEF_DIM, TAC_DIM = 9, 60
STATE_DIM = 2 * EEF_DIM + TAC_DIM


def modality() -> dict:
    state = {"left_wrist_eef": {"start": 0, "end": EEF_DIM}, "right_wrist_eef": {"start": EEF_DIM, "end": 2 * EEF_DIM}}
    off = 2 * EEF_DIM
    for side in ("left", "right"):  # source order: tactile_force.reshape(2, 5, 6) = [side, finger, axis]
        for finger in FINGERS:
            state[f"{side}_{finger}_f6"] = {"start": off, "end": off + 6}
            off += 6
    assert off == STATE_DIM
    return {"state": state}


def pose16_to_eef9(p: np.ndarray) -> np.ndarray:
    """[N,16] row-major 4x4 -> [N,9] = t + R[:,0] + R[:,1] (the columns packing of tianji_sharpa_unified_eef)."""
    M = p.reshape(-1, 4, 4)
    return np.concatenate([M[:, :3, 3], M[:, :3, 0], M[:, :3, 1]], axis=1)


_TREX = None


def _init_worker(trex_repo):
    global _TREX
    sys.path.insert(0, str(Path(trex_repo) / "utils"))
    import unpack_lerobot_to_raw_fk as trex
    _TREX = trex


def build_episode(root, train_shard, dst, ep, idx):
    trex = _TREX
    n = int(ep.length)
    tv = pq.read_table(train_shard / "data/chunk-000" / f"episode_{idx:06d}.parquet")
    st76 = np.stack(tv.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
    assert len(st76) == n, f"ep{idx}: training view has {len(st76)} frames, source {n}"
    tac = trex.episode_arrays(root, ep)["observation.tactile_force"].astype(np.float32)
    state = np.concatenate([pose16_to_eef9(st76[:, 0:16]), pose16_to_eef9(st76[:, 16:32]), tac], axis=1)
    assert state.shape == (n, STATE_DIM) and np.isfinite(state).all()

    cols = {name: tv.column(name) for name in ("timestamp", "frame_index", "episode_index", "index", "task_index")}
    out = dst / "data/chunk-000" / f"episode_{idx:06d}.parquet"
    pq.write_table(pa.table({"observation.state": pa.array(list(state), pa.list_(pa.float32())), **cols}), out)

    path, start = trex.video_location(root, ep, TOP_SRC, FPS)
    mp4 = dst / "videos/chunk-000/top" / f"episode_{idx:06d}.mp4"
    count = 0
    with av.open(str(mp4), "w", format="mp4", options={"movflags": "+faststart"}) as o:
        stream = o.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = TOP_W, TOP_H, "yuv420p"
        for rgb in trex.decode_frames(path, start, n, FPS, "rgb24"):
            assert rgb.shape == (TOP_H, TOP_W, 3), f"unexpected frame shape {rgb.shape}"
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                o.mux(packet)
            count += 1
        for packet in stream.encode():
            o.mux(packet)
    assert count == n, f"ep{idx}: wrote {count} top frames, expected {n}"
    return idx, n


def _worker(job):
    root, train_shard, dst, ep, idx = job
    return build_episode(Path(root), Path(train_shard), Path(dst), pd.Series(ep), idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="v3.0 source dataset root")
    ap.add_argument("--train_view", type=Path, required=True, help="dataset written by convert_trex_v30_to_v21.py")
    ap.add_argument("--dst", type=Path, required=True, help="labeling view to create (its name is the engine task)")
    ap.add_argument("--trex_repo", type=Path, default=DEFAULT_TREX, help="T-Rex checkout (for decode helpers)")
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=0, help="0 = all; N = first N (smoke test, meta left unwritten)")
    args = ap.parse_args()

    root, tv, dst = args.src.resolve(), args.train_view.resolve(), args.dst.resolve()
    shard = tv / "train_shard_0"
    assert not (dst / "meta").exists(), f"{dst} already exists; choose a new --dst"
    manifest = json.loads((tv / "source_manifest.json").read_text())
    tv_eps = [json.loads(line) for line in (shard / "meta/episodes.jsonl").read_text().splitlines()]
    assert [m["episode_index"] for m in manifest] == [e["episode_index"] for e in tv_eps] == list(range(len(tv_eps)))

    src_eps = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))])
    src_eps = src_eps.set_index("episode_index", drop=False)
    jobs = []
    for m, e in zip(manifest, tv_eps):
        ep = src_eps.loc[m["source_episode_index"]]
        assert str(ep.episode_uid) == m["episode_uid"] and int(ep.length) == m["length"] == e["length"]
        jobs.append((str(root), str(shard), str(dst), ep.to_dict(), m["episode_index"]))
    if args.episodes:
        jobs = jobs[: args.episodes]

    for sub in ("meta", "data/chunk-000", "videos/chunk-000/top"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    done = {}
    if args.num_workers == 1:
        _init_worker(args.trex_repo)
        for job in jobs:
            i, n = _worker(job)
            done[i] = n
            print(f"episode {i}: {n} frames ({len(done)}/{len(jobs)})", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker, initargs=(str(args.trex_repo),)) as pool:
            for fut in as_completed([pool.submit(_worker, job) for job in jobs]):
                i, n = fut.result()
                done[i] = n
                print(f"episode {i}: done ({len(done)}/{len(jobs)})", flush=True)
    if args.episodes:
        print(f"smoke test: {len(done)} episodes written, meta skipped -> {dst}")
        return

    tv_info = json.loads((shard / "meta/info.json").read_text())
    scalar = lambda dt: {"dtype": dt, "shape": [1], "names": None}
    names = [f"{k}_{j}" for k, v in modality()["state"].items() for j in range(v["end"] - v["start"])]
    info = {
        **{k: tv_info[k] for k in ("codebase_version", "robot_type", "fps", "chunks_size", "data_path", "video_path",
                                   "total_episodes", "total_frames", "total_tasks", "total_chunks", "splits")},
        "total_videos": len(jobs),
        "features": {
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": names},
            "observation.images.top": {
                "dtype": "video", "shape": [TOP_H, TOP_W, 3], "names": ["height", "width", "channels"],
                "info": {"video.fps": FPS, "video.height": TOP_H, "video.width": TOP_W, "video.channels": 3,
                         "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                         "has_audio": False}},
            "timestamp": scalar("float32"), "frame_index": scalar("int64"), "episode_index": scalar("int64"),
            "index": scalar("int64"), "task_index": scalar("int64"),
        },
    }
    assert sum(done.values()) == info["total_frames"]
    (dst / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    (dst / "meta/modality.json").write_text(json.dumps(modality(), indent=2) + "\n")
    for name in ("episodes.jsonl", "tasks.jsonl"):
        (dst / "meta" / name).write_text((shard / "meta" / name).read_text())
    print(f"done: {len(jobs)} episodes, {sum(done.values())} frames -> {dst}")


if __name__ == "__main__":
    main()
