"""Write the per-shard `meta/stats.json` that LeRobot's MultiLeRobotDataset aggregates.

This is a DIFFERENT file from `dataset_stats_eef64.json`:
  * dataset_stats_eef64.json -- AgenticWAM's 64-D normalizer stats (what actually normalizes actions)
  * meta/stats.json          -- LeRobot's per-shard raw-feature stats, consumed by
                                `aggregate_stats([d.meta.stats for d in datasets])` at dataset
                                construction. Missing it fails with
                                `TypeError: 'NoneType' object is not iterable`.

Schema:
    {feature: {"mean": [...], "std": [...], "min": [...], "max": [...], "count": [N]}}
    observation.state / action  -> lists of length D (the raw vector width)
    scalar columns              -> lists of length 1
    observation.images.<cam>    -> nested [3,1,1] per-channel lists

IMAGE STATS are estimated from a subsample of episodes (default 2 per shard, every 20th frame)
rather than a full decode of all ~300 videos. They are per-channel summaries over frames already
resized to 224x320, and nothing in the AgenticWAM path normalizes with them -- images go through
ToTensor and the model's own [-1,1] scaling. The subsample keeps them honest without spending
~10 minutes decoding for a value that is only aggregated and not applied.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as papq

CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def _stat(x: np.ndarray) -> dict:
    """x: [N, D] -> LeRobot per-feature stat dict."""
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
            "min": x.min(0).tolist(), "max": x.max(0).tolist(), "count": [int(len(x))]}


def image_stats(shard: str, n_eps: int, stride: int) -> dict:
    import av

    out = {}
    for cam in CAMS:
        vids = sorted(glob.glob(os.path.join(shard, f"videos/chunk-000/observation.images.{cam}/*.mp4")))[:n_eps]
        acc = []
        for v in vids:
            c = av.open(v)
            for i, f in enumerate(c.decode(video=0)):
                if i % stride == 0:
                    acc.append(f.to_ndarray(format="rgb24").astype(np.float32).reshape(-1, 3) / 255.0)
            c.close()
        if not acc:
            continue
        a = np.concatenate(acc, 0)          # [P,3] in [0,1]
        out[f"observation.images.{cam}"] = {
            "mean": [[[float(m)]] for m in a.mean(0)],
            "std": [[[float(s)]] for s in a.std(0)],
            "min": [[[float(m)]] for m in a.min(0)],
            "max": [[[float(m)]] for m in a.max(0)],
            "count": [int(len(a))],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True)
    ap.add_argument("--img-eps", type=int, default=2, help="episodes per shard for image stats")
    ap.add_argument("--img-stride", type=int, default=20)
    args = ap.parse_args()

    for shard in sorted(glob.glob(os.path.join(args.dst, "train_shard_*"))):
        files = sorted(glob.glob(os.path.join(shard, "data/chunk-000/*.parquet")))
        st, ac, scal = [], [], {k: [] for k in
                                ("timestamp", "frame_index", "episode_index", "index", "task_index")}
        for f in files:
            t = papq.read_table(f)
            st.append(np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)))
            ac.append(np.stack(t.column("action").to_numpy(zero_copy_only=False)))
            for k in scal:
                scal[k].append(np.asarray(t.column(k).to_numpy(zero_copy_only=False), dtype=np.float64))

        stats = {"observation.state": _stat(np.concatenate(st).astype(np.float64)),
                 "action": _stat(np.concatenate(ac).astype(np.float64))}
        for k, v in scal.items():
            stats[k] = _stat(np.concatenate(v).reshape(-1, 1))
        stats.update(image_stats(shard, args.img_eps, args.img_stride))

        with open(os.path.join(shard, "meta/stats.json"), "w") as fh:
            json.dump(stats, fh)
        print(f"  {os.path.basename(shard)}: {len(files)} eps, "
              f"{stats['action']['count'][0]} frames, {len(stats)} features", flush=True)

    print("LEROBOT_STATS_DONE", flush=True)


if __name__ == "__main__":
    main()
