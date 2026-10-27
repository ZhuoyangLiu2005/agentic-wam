"""Compute unified-64D statistics for one SFT dataset or an entire pretraining mixture.

Shared pretraining mode (all actual training sources, one output file):
    python utils/make_stats.py --task pretrain_magic_soup \
        data.yam_root=... data.ego_root=... data.shared_norm_stats=.../dataset_stats_eef64.json

This mode pools transformed values before computing statistics. Moments/extrema
cover every scanned value; quantiles use at most --max-samples observations or
action windows (default 100000). --max-samples 0 requests exact pooled quantiles
and retains all transformed values in memory. --stride selects action-window starts.

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
from pathlib import Path
import tempfile

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
    ap.add_argument("--dst", help="converted SFT dataset root (contains train_shard_*)")
    ap.add_argument("--transform",
                    help="raw -> unified 64-D transform: yam | tianji_sharpa | a dotted class path")
    ap.add_argument("--task", help="pretraining task: compute ONE set of stats over its training sources")
    ap.add_argument("--max-samples", type=int, default=100_000,
                    help="shared mode: quantile reservoir size; 0 keeps all values for exact quantiles")
    ap.add_argument("--seed", type=int, default=42, help="shared mode: quantile reservoir seed")
    ap.add_argument("--stride", type=int, default=5, help="window stride when sampling chunks")
    ap.add_argument("--out", default=None)
    args, overrides = ap.parse_known_args()

    if args.task:
        if args.dst or args.transform:
            ap.error("--task cannot be combined with --dst/--transform")
        from hydra import compose, initialize_config_dir
        from agenticwam.datasets.lerobot.shared_stats import compute_shared_stats
        from agenticwam.utils.config_resolvers import register_default_resolvers

        register_default_resolvers()
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
            cfg = compose(config_name="train", overrides=[f"task={args.task}", *overrides])
            out = Path(args.out or str(cfg.data.shared_norm_stats)).expanduser()
            stats = compute_shared_stats(cfg.data, stride=args.stride,
                                         max_samples=args.max_samples, seed=args.seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a complete JSON; failed scans must not replace a valid file.
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=out.parent, suffix=".tmp", delete=False) as f:
                temp_path = Path(f.name)
                json.dump(stats, f, allow_nan=False)
            os.replace(temp_path, out)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        print(f"[shared stats] wrote {out}: {stats['num_transition']} frames, "
              f"{len(stats['shared_normalization']['sources'])} sources", flush=True)
        print("STATS_DONE", flush=True)
        return

    if not args.dst or not args.transform or overrides:
        ap.error("use --dst <root> --transform <name>, or --task <task> [Hydra overrides]")

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
