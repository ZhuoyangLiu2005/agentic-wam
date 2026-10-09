#!/usr/bin/env python3
"""Checks to run before a downstream SFT launch. Called by the `train_*.sh` entry points.

    python scripts/preflight_sft.py --task <task> --robot yam|tianji_sharpa|robotwin [--nproc 8] [hydra overrides ...]

Each check exists because its failure trains silently on wrong data instead of crashing:

1. config      the task composes exactly as `scripts/train.py` will compose it (same overrides)
2. robot       the data processor uses the robot's 64-D transform (a YAM task with the Sharpa
               transform, or none, still trains -- on garbage actions)
3. paths       every absolute path in the composed config exists
4. code cache  `policy_code_idx_path` covers every frame of `dataset_dirs` (frame_offset + n_frames);
               a short cache shifts every label onto the wrong frame
5. instructions every episode's SERVED instruction (parquet task_index -> tasks.jsonl) matches
               meta/episodes.jsonl and has a precomputed Cosmos text embed (real-robot only)
6. GPUs        the first --nproc GPUs are idle

Exits non-zero on any failure; prints one line per check.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "utils"))

TRANSFORMS = {
    "yam": "YamUnifiedEEFTransform",
    "tianji_sharpa": "TianjiSharpaUnifiedEEFTransform",
    "robotwin": None,            # RoboTwin configs choose their own action space (14-D joint or 64-D EEF)
}
SKIP_PATH_KEYS = {"output_dir"}  # written by the run, not read


class Report:
    def __init__(self):
        self.failed = False

    def ok(self, name, msg):
        print(f"[ok]   {name:12s} {msg}", flush=True)

    def warn(self, name, msg):
        print(f"[warn] {name:12s} {msg}", flush=True)

    def fail(self, name, msg):
        print(f"[FAIL] {name:12s} {msg}", flush=True)
        self.failed = True


def compose(task, overrides):
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    from agenticwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return hydra_compose(config_name="train", overrides=[f"task={task}", *overrides])


def walk_paths(node, prefix=""):
    """Yield (key, value) for every absolute-path string leaf; unresolvable leaves are skipped."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(node, DictConfig):
        for k in node.keys():
            try:
                v = node[k]
            except Exception:                                    # noqa: BLE001 (e.g. ${now:} outside hydra)
                continue
            yield from walk_paths(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(node, ListConfig):
        for i in range(len(node)):
            try:
                v = node[i]
            except Exception:                                    # noqa: BLE001
                continue
            yield from walk_paths(v, f"{prefix}[{i}]")
    elif isinstance(node, str) and node.startswith("/") and " " not in node:
        yield prefix, node


def episode_lengths(shard):
    return [int(json.loads(l)["length"]) for l in open(os.path.join(shard, "meta", "episodes.jsonl")) if l.strip()]


def check_code_cache(r, split, d):
    path = d.get("policy_code_idx_path")
    if not path:
        return
    import torch

    dirs = list(d.get("dataset_dirs") or [])
    if not all(os.path.isfile(os.path.join(x, "meta", "episodes.jsonl")) for x in dirs):
        r.warn("code cache", f"{split}: dataset_dirs are not all LeRobot shards; coverage not checked")
        return
    sel = d.get("episode_indices")
    if sel is None:
        n = sum(sum(episode_lengths(x)) for x in dirs)
        what = "dataset frames"
    else:
        # the loader indexes the cache by its own enumeration, i.e. over the SELECTED episodes only
        if isinstance(sel, str):
            sel = json.load(open(sel))
        if isinstance(sel, dict) or len(dirs) != 1:
            r.warn("code cache", f"{split}: per-subset episode_indices; coverage not checked")
            return
        lens = {int(json.loads(l)["episode_index"]): int(json.loads(l)["length"])
                for l in open(os.path.join(dirs[0], "meta", "episodes.jsonl")) if l.strip()}
        n = sum(lens[int(i)] for i in sel)
        what = f"frames in {len(sel):,} selected episodes"
    off = int(d.get("frame_offset") or 0)
    ids = torch.load(path, map_location="cpu", weights_only=False)["code_id_by_idx"]
    if int(ids.numel()) < off + n:
        r.fail("code cache", f"{split}: {Path(path).name} has {int(ids.numel()):,} frames < frame_offset {off:,} + "
                             f"{n:,} {what}")
        return
    empty = int((ids[off:off + n] == 0).sum())
    msg = f"{split}: {Path(path).name} covers {n:,} frames (offset {off:,}); {empty:,} without code"
    if empty == n:
        r.fail("code cache", msg + " -- every frame is empty")
    elif int(ids.numel()) != off + n and off == 0:
        r.warn("code cache", msg + f"; cache is longer ({int(ids.numel()):,}) -- check the episode order")
    elif empty:
        r.warn("code cache", msg)
    else:
        r.ok("code cache", msg)


def check_instructions(r, split, d):
    from verify_instruction_embeds import check

    cache = d.get("cosmos_text_cache_dir")
    if not cache:
        r.warn("instructions", f"{split}: no cosmos_text_cache_dir; not checked")
        return
    census, bad, miss = check(list(d.get("dataset_dirs") or []), cache)
    msg = f"{split}: {sum(census.values())} episodes, {len(census)} instructions"
    if bad:
        r.fail("instructions", f"{msg}; {len(bad)} episodes serve a different instruction than meta, e.g. {bad[0]}")
    if miss:
        r.fail("instructions", f"{msg}; {len(set(miss))} instructions have no text embed, e.g. {sorted(set(miss))[0]!r} "
                               f"(run scripts/precompute_cosmos_text_embeds.py)")
    if not (bad or miss):
        r.ok("instructions", msg)


def check_gpus(r, nproc):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        r.fail("gpus", f"nvidia-smi failed: {e}")
        return
    rows = [tuple(int(x) for x in l.split(",")) for l in out.strip().splitlines()]
    if len(rows) < nproc:
        r.fail("gpus", f"{len(rows)} GPUs visible, --nproc {nproc}")
        return
    busy = [i for i, mem in rows[:nproc] if mem > 2000]
    if busy:
        r.fail("gpus", f"GPUs {busy} already hold >2 GB")
    else:
        r.ok("gpus", f"{nproc} idle")


def dataset_nodes(node, label):
    """Yield (label, node) for every dataset block (a mapping with `dataset_dirs`) under `node`, at any depth --
    a split may be one dataset, or a ConcatDataset / wrapper holding several."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(node, DictConfig):
        if "dataset_dirs" in node:
            yield label, node
            return
        for k in node.keys():
            try:
                v = node[k]
            except Exception:                                    # noqa: BLE001
                continue
            if isinstance(v, (DictConfig, ListConfig)):
                yield from dataset_nodes(v, label if k in ("datasets", "dataset") else f"{label}.{k}")
    elif isinstance(node, ListConfig):
        for i in range(len(node)):
            yield from dataset_nodes(node[i], f"{label}[{i}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--robot", required=True, choices=sorted(TRANSFORMS))
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--skip-gpu-check", action="store_true")
    a, overrides = ap.parse_known_args()
    task = a.task[:-5] if a.task.endswith(".yaml") else a.task
    r = Report()

    try:
        cfg = compose(task, overrides)
    except Exception as e:                                       # noqa: BLE001
        r.fail("config", f"task config {task!r} does not compose: {type(e).__name__}: {e}")
        sys.exit(1)
    steps, bs = cfg.get("max_steps"), cfg.get("batch_size")
    r.ok("config", f"{task}: max_steps={steps} batch_size/gpu={bs}")

    nodes = [(lbl, d) for s in ("train", "val") if s in cfg.data and cfg.data[s] is not None
             for lbl, d in dataset_nodes(cfg.data[s], s)]
    if not nodes:
        r.fail("config", "no dataset block (a mapping with dataset_dirs) under data.train / data.val")
        sys.exit(1)
    want = TRANSFORMS[a.robot]
    for lbl, d in nodes:
        proc = d.get("processor")
        names = [str(t.get("_target_", "")).rsplit(".", 1)[-1]
                 for t in ((proc.get("action_state_transforms") if proc is not None else None) or [])]
        others = [t for t in TRANSFORMS.values() if t and t != want and t in names]
        if want and want not in names:
            r.fail("robot", f"{lbl}: expected {want}, processor has {names or 'no transform'}")
        elif want and others:                                    # RoboTwin may reuse a real robot's 64-D transform
            r.fail("robot", f"{lbl}: {others} is another robot's transform")
        else:
            r.ok("robot", f"{lbl}: {names or 'no transform'}")

    missing = [(k, v) for k, v in walk_paths(cfg) if k.rsplit(".", 1)[-1] not in SKIP_PATH_KEYS and not os.path.exists(v)]
    for k, v in missing:
        r.fail("paths", f"{k} = {v} does not exist")
    if not missing:
        r.ok("paths", "every absolute path exists")
    else:
        sys.exit(1)                                              # later checks would only re-report it

    for lbl, d in nodes:
        check_code_cache(r, lbl, d)
        if a.robot != "robotwin":
            check_instructions(r, lbl, d)

    if a.robot != "robotwin" and steps and bs:
        n = sum(sum(episode_lengths(x)) for lbl, d in nodes if lbl.startswith("train") for x in d.dataset_dirs
                if os.path.isfile(os.path.join(x, "meta", "episodes.jsonl")))
        if n:
            print(f"       {'schedule':12s} {steps:,} steps x {bs * a.nproc * int(os.environ.get('NNODES', 1))} "
                  f"samples = {steps * bs * a.nproc * int(os.environ.get('NNODES', 1)) / n:.1f} epochs of {n:,} frames")

    if a.skip_gpu_check:
        r.warn("gpus", "skipped")
    else:
        check_gpus(r, a.nproc)

    if not (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") in ("offline", "disabled")):
        r.warn("wandb", "WANDB_API_KEY is not set; export it, or WANDB_MODE=offline")

    sys.exit(1 if r.failed else 0)


if __name__ == "__main__":
    main()
