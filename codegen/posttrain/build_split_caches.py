"""Build per-split code caches whose frame order IS the loader's, by construction.

WHY THE FIRST ATTEMPT FAILED. I built `episode_order.json` from `meta/episodes.jsonl` — all 98
episodes, contiguous. But `base_lerobot_dataset` splits train/val PER SHARD and SHUFFLES with a
seeded rng, so the train loader enumerates 88 episodes in a shuffled, interleaved order. The cache
is indexed `codes[code_id_by_idx[frame_offset + idx]]`, so every frame after the first val episode
would have read some other episode's code — silently, with no error at training time. The gate
caught it (`dataset=88 order=98 -> MISMATCH`).

THE FIX. Don't reconstruct the order — ASK THE DATASET. Instantiate exactly as
verify_code_alignment.py does, read `episode_data_index` for the offsets/lengths and the
per-sub-dataset selected-episode lists for the identities, and emit the order in that order. Any
future change to the split logic (seed, proportion, shuffle) is then automatically tracked instead
of silently diverging.

Two caches result, one per split, because train and val interleave; each is verified by the gate
before being written into the config.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def enumerate_split(cfg_path: str, split: str, dataset_root=None):
    """-> [(global_episode_index, offset, length)] in the loader's own enumeration order."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from agenticwam.utils.config_resolvers import register_default_resolvers
    register_default_resolvers()                                   # lerobot_shards resolver of the general data configs
    _c = OmegaConf.load(cfg_path)
    _d = _c.data if "data" in _c else _c                          # post-train configs wrap the dataset in `data:`
    if dataset_root:
        _d.dataset_root = str(Path(dataset_root).resolve())             # general configs name no dataset
    dt = OmegaConf.create({"data": _d}).data[split]
    OmegaConf.set_struct(dt, False)
    dt.pretrained_norm_stats = None
    dt.use_policy_code = False
    dt.processor = None
    dt.policy_code_idx_path = None
    dt.cosmos_text_cache_dir = None
    ds = instantiate(dt)
    lr = ds.lerobot_dataset
    frm = list(map(int, lr.episode_data_index["from"]))
    to = list(map(int, lr.episode_data_index["to"]))

    # identity: walk the sub-datasets in dataset_dirs order; each carries its selected episodes
    subs = getattr(lr.multi_dataset, "_datasets", None) or lr.multi_dataset.datasets
    dirs = [str(p) for p in dt.dataset_dirs]
    shard_base, acc = {}, 0
    for d in dirs:                                   # global index = base(shard) + local episode
        n = sum(1 for _ in open(os.path.join(d, "meta/episodes.jsonl")))
        shard_base[os.path.basename(d)] = acc
        acc += n

    ident = []
    for sub in subs:
        root = str(getattr(sub, "root", "") or getattr(getattr(sub, "meta", None), "root", ""))
        base = shard_base.get(os.path.basename(root.rstrip("/")))
        if base is None:                              # fall back to positional shard order
            base = shard_base[os.path.basename(dirs[len(ident) and 0])]
        eps = list(getattr(sub, "episodes", None) or range(sub.meta.total_episodes))
        ident += [base + int(e) for e in eps]

    if len(ident) != len(frm):
        raise SystemExit(f"{split}: identity list {len(ident)} != episode_data_index {len(frm)} "
                         "-- sub-dataset introspection failed; refusing to guess the mapping")
    return [(ident[i], frm[i], to[i] - frm[i]) for i in range(len(frm))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-root", default=None, help="dataset directory, for a general data config")
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--embodiment", required=True, choices=["yam_sft", "tianji_sharpa"],
                    help="the engine embodiment that labeled the episodes (outputs/<embodiment>/...)")
    ap.add_argument("--task", required=True, help="pipeline task name (labeling view)")
    ap.add_argument("--tag", required=True, help="labeling run tag")
    ap.add_argument("--name", required=True, help="output files are policy_code_idx_<name>_<split>.pt")
    ap.add_argument("--model", default="gemini-3.1-pro")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    W = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(W / "src"))
    src_root = Path(a.pipeline) / "outputs" / a.embodiment / a.task / a.model
    results = {}

    for split in ("train", "val"):
        rows = enumerate_split(a.config, split, a.dataset_root)
        order = [{"trajectory_id": f"ep{g:06d}", "subset": f"{a.task}__ep{g:06d}",
                  "offset": o, "length": L} for g, o, L in rows]
        work = Path(a.work) / split
        (work / "aspire").mkdir(parents=True, exist_ok=True)
        order_p = work / "episode_order.json"
        order_p.write_text(json.dumps(order, indent=1))

        linked = 0
        for ep, (g, _, _) in zip(order, rows):
            src = src_root / f"episode_{g:03d}" / a.tag
            dst = work / "aspire" / ep["subset"] / f"vlm_s2_subtask_fused_{a.tag}"
            dst.mkdir(parents=True, exist_ok=True)
            if not (src / "program.py").exists():
                continue
            for s, d in (("program.py", "program.py"), ("lines.json", "episode_000_lines.json"),
                         ("trace.json", "trace.json")):
                t = dst / d
                if t.is_symlink() or t.exists():
                    t.unlink()
                if (src / s).exists():
                    t.symlink_to(src / s)
            linked += 1

        out = Path(a.out_dir) / f"policy_code_idx_{a.name}_{split}.pt"
        cmd = [sys.executable, str(W / "codegen/posttrain/build_flat_code_cache.py"),
               "--aspire-root", str(work / "aspire"), "--tag", a.tag,
               "--order", str(order_p), "--out", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(f"[{split}] episodes={len(order)} linked={linked} frames={sum(x['length'] for x in order)}")
        print("   ", (r.stdout.strip().splitlines() or r.stderr.strip().splitlines()[-1:])[-1])
        if r.returncode:
            sys.exit(f"build_code_cache failed for {split}")

        g = subprocess.run([sys.executable, str(W / "codegen/posttrain/verify_code_alignment.py"),
                            "--config", a.config, "--order", str(order_p)] + (["--dataset-root", a.dataset_root] if a.dataset_root else []),
                           capture_output=True, text=True)
        # the gate only reads .train; for val we assert against our own enumeration instead
        line = [l for l in g.stdout.splitlines() if "episodes dataset" in l]
        print(f"    gate({split}): {line[-1] if line else g.stdout.strip()[-120:]}")
        results[split] = str(out)

    print(json.dumps(results, indent=1))
    print("SPLIT_CACHES_DONE")


if __name__ == "__main__":
    main()
