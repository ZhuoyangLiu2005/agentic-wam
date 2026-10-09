"""Compute the 64-D `dataset_stats_eef64.json` for a converted SFT dataset (docs/DATA.md §3).

Schema:
    {state,action}.default.{global,stepwise}_{min,max,q01,q99,mean,std}
      * state  stepwise lists have length 1   (state is a single frame)
      * action stepwise lists have length 32  (the action-chunk horizon)
    num_episodes, num_transition

Stats are computed on the **64-D transformed** values, not the raw vectors, because that is what
the model consumes and what the normalizer is applied to. --transform picks the raw -> 64-D
transform: a shortcut (`yam`, `tianji_sharpa`, `ego`) or a dotted class path.

ACTION WINDOWING. For a window starting at frame t: base = state[t], chunk = action[t : t+32]
(`action[t]` is the command issued at t, targeting t+1).

Constant-zero dims (the gripper slots on a hand robot, the hand block on a gripper robot) come out
with q01 == q99 == 0. That is expected and the normalizer's ignore-dim path handles it -- do NOT
clamp them to a fake range.

Run with PYTHONPATH=<repo>/src.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as papq
import torch

import importlib

_SHORTCUTS = {
    "yam": "agenticwam.datasets.lerobot.transforms.yam_unified_eef.YamUnifiedEEFTransform",
    "tianji_sharpa": "agenticwam.datasets.lerobot.transforms.tianji_sharpa_unified_eef.TianjiSharpaUnifiedEEFTransform",
    "ego": "agenticwam.datasets.lerobot.transforms.ego_unified_eef.EgoUnifiedEEFTransform",
}


def _load_transform(name: str):
    path = _SHORTCUTS.get(name, name)
    mod, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)(keys=["default"])

HORIZON = 32


def _stats(x: np.ndarray) -> dict:
    """x: [N, D] -> the six global aggregates."""
    return {
        "min": x.min(0).tolist(), "max": x.max(0).tolist(),
        "q01": np.percentile(x, 1, axis=0).tolist(), "q99": np.percentile(x, 99, axis=0).tolist(),
        "mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", required=True, help="converted dataset root (contains train_shard_*)")
    ap.add_argument("--transform", required=True,
                    help="raw -> unified 64-D transform: yam | tianji_sharpa | a dotted class path")
    ap.add_argument("--stride", type=int, default=5, help="window stride when sampling chunks")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tf = _load_transform(args.transform)
    files = sorted(glob.glob(os.path.join(args.dst, "train_shard_*/data/chunk-000/*.parquet")))
    if not files:
        raise SystemExit(f"no parquet under {args.dst}")

    state_all, act_steps = [], []
    total_frames = 0
    for f in files:
        t = papq.read_table(f)
        st = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
        ac = np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
        total_frames += len(st)

        s64 = tf._state64(torch.from_numpy(st)).numpy()
        state_all.append(s64)

        starts = np.arange(0, max(len(st) - HORIZON, 0), args.stride)
        if len(starts) == 0:
            continue
        base = torch.from_numpy(st[starts])                                  # [W,76]
        chunk = torch.from_numpy(np.stack([ac[t0:t0 + HORIZON] for t0 in starts]))   # [W,32,76]
        act_steps.append(tf._action64(chunk, base).numpy())                  # [W,32,64]
        print(f"  {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))}/"
              f"{os.path.basename(f)}  T={len(st)} windows={len(starts)}", flush=True)

    S = np.concatenate(state_all, 0)                     # [N,64]
    A = np.concatenate(act_steps, 0)                     # [W,32,64]
    print(f"[stats] state frames {S.shape}, action windows {A.shape}", flush=True)

    sg = _stats(S)
    ag = _stats(A.reshape(-1, A.shape[-1]))
    per_step = [_stats(A[:, k]) for k in range(HORIZON)]

    def pack(g, steps):
        out = {f"global_{k}": g[k] for k in ("min", "max", "q01", "q99", "mean", "std")}
        for k in ("min", "max", "q01", "q99", "mean", "std"):
            out[f"stepwise_{k}"] = [s[k] for s in steps]
        return {"default": out}

    stats = {"state": pack(sg, [sg]), "action": pack(ag, per_step),
             "num_episodes": len(files), "num_transition": int(total_frames)}
    out = args.out or os.path.join(args.dst, "dataset_stats_eef64.json")
    json.dump(stats, open(out, "w"))
    print(f"[stats] wrote {out}", flush=True)

    z = [i for i in range(64) if sg["q01"][i] == sg["q99"][i] == 0.0]
    print(f"[stats] constant-zero state dims: {z}", flush=True)
    print("STATS_DONE", flush=True)


if __name__ == "__main__":
    main()
