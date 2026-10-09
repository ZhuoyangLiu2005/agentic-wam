#!/usr/bin/env python3
"""Export one collection (save_dir with data/, alignment/, manifest.jsonl) to a LeRobot v2.1 dataset.

    python -m capsim.data.lerobot_export --raw-root <save_dir> --out <dataset>/robotwin2.0 [--work DIR] [--workers 8]

Kept episodes = manifest rows with success == true sorted by seed (episode_index = rank; the slicing engine orders
the same way). Per-frame instruction = the alignment JSON's filled `instructions.seen` cycled over frames (t % n),
else the template `instruction`. Re-running skips finished episodes.
"""
import argparse
import json
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from capsim.data.hdf5 import (ACTION_KEY, CAM_MAP, EMBODIMENT, FPS, IMG_KEYS, PROFILE, STATE_KEY, VIDEO_H, VIDEO_W,  # noqa: E402
                              decode_upscale, downstream_root, encode_frames, read_hdf5_episode)
from capsim.data.instructions import reconcile  # noqa: E402

TEMPLATE = Path(__file__).with_name("lerobot_info_template.json")
A = None


def check_template(info=None):
    """The info.json template must agree with the embodiment profile (robot_type, fps, joint names, video size), so
    the two can never drift; raises AssertionError naming the field otherwise."""
    info = info or json.load(open(TEMPLATE))
    rec = PROFILE.recording
    names = list(rec.joint_names)
    checks = {
        "robot_type": (info["robot_type"], str(rec.robot_type)),
        "fps": (info["fps"], int(rec.fps)),
        f"features.{STATE_KEY}.shape": (info["features"][STATE_KEY]["shape"], [int(rec.state_dim)]),
        f"features.{STATE_KEY}.names": (info["features"][STATE_KEY]["names"], [names]),
        f"features.{ACTION_KEY}.names": (info["features"][ACTION_KEY]["names"], [names]),
        "state_dim vs joint_names": (len(names), int(rec.state_dim)),
    }
    for k in IMG_KEYS:
        f = info["features"][k]
        checks[f"features.{k}.shape"] = (f["shape"], [VIDEO_H, VIDEO_W, 3])
        checks[f"features.{k}.info.video.fps"] = (f["info"]["video.fps"], FPS)
        checks[f"features.{k}.info.video.height/width"] = ((f["info"]["video.height"], f["info"]["video.width"]), (VIDEO_H, VIDEO_W))
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    assert not bad, f"{TEMPLATE.name} disagrees with profile {EMBODIMENT!r}: " + "; ".join(
        f"{k}: template {a!r} != profile {b!r}" for k, (a, b) in bad.items())
    return info


def kept_episodes(raw_root):
    rows = [json.loads(l) for l in open(Path(raw_root) / "manifest.jsonl") if l.strip()]
    keep = sorted((r for r in rows if r.get("success")), key=lambda r: int(r["seed"]))
    return [(int(r["seed"]), int(r["num_frames"])) for r in keep]


def episode_instructions(raw_root, seed, task_name=None, n_max=None):
    """Filled paraphrases for the episode (the first `n_max` of the recorded list when set), arm placeholders
    reconciled with the program's own arm; falls back to the unfilled template. Returns (list, changed)."""
    al = json.load(open(Path(raw_root) / "alignment" / f"episode{seed}.json"))
    task_name = task_name or _task_name(raw_root)
    _, ins, changed = reconcile(task_name, al) if task_name else (None, al.get("instructions") or {}, False)
    seen = ins.get("seen") or []
    if n_max:
        seen = seen[:n_max]
    return (seen if seen else [al.get("instruction") or "robotwin task"]), changed


def _task_name(raw_root):
    rows = [json.loads(l) for l in open(Path(raw_root) / "manifest.jsonl") if l.strip()]
    for r in rows:
        if r.get("task_name"):
            return r["task_name"]
    cfg = Path(raw_root).parent / "collect.yaml"          # make_dataset.py layout
    if cfg.exists():
        import re
        m = re.search(r"task_name:\s*(\S+)", cfg.read_text())
        return m.group(1) if m else None
    return None


def _phase1_episode(ep_i):
    seed, n_manifest = A.eps[ep_i]
    ep_work = A.work / f"ep{ep_i:06d}"
    if (ep_work / "arrays.npz").exists() and (ep_work / "stats.json").exists() and all(
            (A.out / f"videos/chunk-000/{k}/episode_{ep_i:06d}.mp4").exists() for k in IMG_KEYS):
        # frames/stats are reusable, but instr.json depends on --n-instructions and the alignment JSON: rewrite it
        T = int(np.load(ep_work / "arrays.npz")["length"])
        seen, changed = episode_instructions(A.raw_root, seed, A.task_name, A.n_instructions)
        json.dump({"per_frame": [seen[t % len(seen)] for t in range(T)], "unique": list(dict.fromkeys(seen)), "seed": seed,
                   "arm_reconciled": changed}, open(ep_work / "instr.json", "w"))
        return ep_i, -1
    downstream_root()
    from agenticwam.datasets.lerobot.lerobot.datasets.compute_stats import compute_episode_stats
    from agenticwam.datasets.lerobot.lerobot.datasets.utils import serialize_dict
    state, action, cams, T = read_hdf5_episode(Path(A.raw_root) / "data" / f"episode{seed}.hdf5")
    assert T == n_manifest, f"ep{ep_i} seed{seed}: hdf5 T={T} != manifest num_frames={n_manifest}"
    ep_work.mkdir(parents=True, exist_ok=True)
    features = json.load(open(TEMPLATE))["features"]
    episode_data = {STATE_KEY: state, ACTION_KEY: action}
    for _, key in CAM_MAP:
        img_dir = ep_work / key; img_dir.mkdir(parents=True, exist_ok=True); paths = []
        for t in range(T):
            fp = img_dir / f"frame_{t:06d}.jpeg"; decode_upscale(cams[key][t]).save(fp, format="JPEG", quality=95); paths.append(str(fp))
        encode_frames(img_dir, A.out / f"videos/chunk-000/{key}/episode_{ep_i:06d}.mp4")
        episode_data[key] = paths
    ts = (np.arange(T, dtype=np.float32) / FPS).reshape(T, 1); fidx = np.arange(T, dtype=np.int64).reshape(T, 1)
    episode_data.update({"timestamp": ts, "frame_index": fidx, "episode_index": np.full((T, 1), ep_i, dtype=np.int64),
                         "index": np.zeros((T, 1), dtype=np.int64), "task_index": np.zeros((T, 1), dtype=np.int64)})
    stats = compute_episode_stats(episode_data, features, is_compute_episode_stats_image=True)
    seen, changed = episode_instructions(A.raw_root, seed, A.task_name, A.n_instructions)
    np.savez(ep_work / "arrays.npz", state=state, action=action, timestamp=ts.astype(np.float32), frame_index=fidx, length=np.int64(T))
    json.dump({"per_frame": [seen[t % len(seen)] for t in range(T)], "unique": list(dict.fromkeys(seen)), "seed": seed,
               "arm_reconciled": changed}, open(ep_work / "instr.json", "w"))
    json.dump(serialize_dict(stats), open(ep_work / "stats.json", "w"))
    for _, key in CAM_MAP:
        shutil.rmtree(ep_work / key, ignore_errors=True)
    return ep_i, T


def phase1(workers):
    A.work.mkdir(parents=True, exist_ok=True); (A.out / "videos/chunk-000").mkdir(parents=True, exist_ok=True)
    eps = list(range(len(A.eps)))
    if workers <= 1:
        for e in eps:
            print("ph1", *_phase1_episode(e), flush=True)
        return
    with mp.Pool(workers) as pool:
        for ep_i, T in pool.imap_unordered(_phase1_episode, eps):
            print(f"ph1 ep{ep_i} T={T}", flush=True)


def phase2():
    import pyarrow as pa
    import pyarrow.parquet as pq
    downstream_root()
    from agenticwam.datasets.lerobot.lerobot.datasets.utils import (cast_stats_to_numpy, write_episode, write_episode_stats,
                                                                  write_info, write_task)
    info = check_template()
    (A.out / "meta").mkdir(parents=True, exist_ok=True); (A.out / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    for m in ["tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl", "info.json"]:
        (A.out / "meta" / m).unlink(missing_ok=True)
    task_to_index, global_index, total_frames, reconciled = {}, 0, 0, []
    for ep_i in range(len(A.eps)):
        ep_work = A.work / f"ep{ep_i:06d}"; arr = np.load(ep_work / "arrays.npz")
        instr = json.load(open(ep_work / "instr.json")); stats = cast_stats_to_numpy(json.load(open(ep_work / "stats.json")))
        T = int(arr["length"])
        if instr.get("arm_reconciled"):
            reconciled.append(instr["seed"])
        for task in instr["unique"]:
            if task not in task_to_index:
                task_to_index[task] = len(task_to_index); write_task(task_to_index[task], task, A.out)
        per_frame_ti = np.array([task_to_index[s] for s in instr["per_frame"]], dtype=np.int64)
        index_col = np.arange(global_index, global_index + T, dtype=np.int64)
        pq.write_table(pa.table({
            STATE_KEY: pa.array([r.tolist() for r in arr["state"]], type=pa.list_(pa.float32())),
            ACTION_KEY: pa.array([r.tolist() for r in arr["action"]], type=pa.list_(pa.float32())),
            "timestamp": pa.array(arr["timestamp"].reshape(-1).tolist(), type=pa.float32()),
            "frame_index": pa.array(arr["frame_index"].reshape(-1).tolist(), type=pa.int64()),
            "episode_index": pa.array([ep_i] * T, type=pa.int64()),
            "index": pa.array(index_col.tolist(), type=pa.int64()),
            "task_index": pa.array(per_frame_ti.tolist(), type=pa.int64()),
        }), A.out / f"data/chunk-000/episode_{ep_i:06d}.parquet")
        write_episode({"episode_index": ep_i, "tasks": list(dict.fromkeys(instr["per_frame"])), "length": T}, A.out)
        for k, col in (("index", index_col), ("task_index", per_frame_ti)):
            stats[k] = {"min": np.array([float(col.min())]), "max": np.array([float(col.max())]), "mean": np.array([float(col.mean())]),
                        "std": np.array([float(col.std())]), "count": np.array([T])}
        write_episode_stats(ep_i, stats, A.out); global_index += T; total_frames += T
    N = len(A.eps)
    info.update({"total_episodes": N, "total_frames": total_frames, "total_tasks": len(task_to_index), "total_videos": N * len(IMG_KEYS),
                 "total_chunks": 1, "splits": {"train": f"0:{N}"}})
    for k in IMG_KEYS:
        info["features"][k]["shape"] = [VIDEO_H, VIDEO_W, 3]; info["features"][k]["info"]["video.height"] = VIDEO_H; info["features"][k]["info"]["video.width"] = VIDEO_W
    write_info(info, A.out)
    json.dump({"raw_root": str(A.raw_root), "episodes": [{"episode_index": i, "seed": s, "num_frames": n} for i, (s, n) in enumerate(A.eps)]},
              open(A.out / "meta" / "episode_seed_map.json", "w"), indent=1)
    json.dump({"arm_placeholder_reconciled_seeds": reconciled}, open(A.out / "meta" / "instruction_reconcile.json", "w"))
    print(f"DONE: {N} episodes, {total_frames} frames, {len(task_to_index)} distinct instructions -> {A.out} "
          f"({len(reconciled)} episodes had the arm placeholder corrected to the program's arm)", flush=True)


def main():
    global A
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", required=True, help="collection save_dir (data/, alignment/, manifest.jsonl)")
    ap.add_argument("--out", required=True, help="LeRobot dataset root to write (…/robotwin2.0)")
    ap.add_argument("--work", default=None, help="scratch for per-episode frames/stats (default <out>/../_work)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--phase", choices=["all", "1", "2"], default="all")
    ap.add_argument("--task-name", default=None, help="RoboTwin task (default: from manifest / collect.yaml) — for instruction regeneration")
    ap.add_argument("--n-instructions", type=int, default=None,
                    help="keep only the first N recorded paraphrases per episode (default: all); bounds the trainer's text cache")
    A = ap.parse_args()
    A.raw_root = Path(A.raw_root); A.out = Path(A.out); A.work = Path(A.work or A.out.parent / "_work")
    check_template()
    A.eps = kept_episodes(A.raw_root)
    A.task_name = A.task_name or _task_name(A.raw_root)
    if not A.eps:
        sys.exit(f"no successful episodes in {A.raw_root}/manifest.jsonl")
    print(f"{len(A.eps)} kept episodes, {sum(n for _, n in A.eps)} frames -> {A.out}", flush=True)
    if A.phase in ("all", "1"):
        phase1(A.workers)
    if A.phase in ("all", "2"):
        phase2()


if __name__ == "__main__":
    main()
