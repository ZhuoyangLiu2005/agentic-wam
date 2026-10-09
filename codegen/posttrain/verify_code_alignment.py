#!/usr/bin/env python3
"""GATE: assert the loader's frame enumeration == episode_order.json used to build the code cache.

Instantiates the dataset from a data config, reads its episode_data_index (from/to per episode), and
checks the per-episode (offset,length) match episode_order. Must pass before training on the cache.

    # post-training: a general per-robot data config
    python codegen/posttrain/verify_code_alignment.py --config configs/real_post_train/data/yam.yaml \
        --dataset-root <dataset> --order <dataset>/episode_order.json
    # pretraining: one source of the pretraining data config
    python codegen/posttrain/verify_code_alignment.py --config configs/real_pre_train/data/pretrain_magic_soup.yaml \
        --source ego --dataset-root <ego_root> --order <ego_root>/episode_order.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="a data config (post-train: uses .train; pretrain: uses .sources.<source>)")
    ap.add_argument("--order", required=True)
    ap.add_argument("--dataset-root", default=None, help="dataset directory, for a general data config")
    ap.add_argument("--source", default=None, choices=["yam", "ego"], help="pretraining source in configs/real_pre_train/data/")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from agenticwam.utils.config_resolvers import register_default_resolvers
    register_default_resolvers()                                   # lerobot_shards resolver of the general data configs
    _c = OmegaConf.load(args.config)
    _d = _c.data if "data" in _c else _c                          # post-train configs wrap the dataset in `data:`
    if args.source:
        if args.dataset_root:
            _d[f"{args.source}_root"] = str(Path(args.dataset_root).resolve())
        for s in ("yam", "ego"):                                  # the other source's ??? must not block resolution
            if OmegaConf.is_missing(_d, f"{s}_root"):
                _d[f"{s}_root"] = "/unused"
        dt = OmegaConf.create({"data": _d}).data.sources[args.source]
    else:
        if args.dataset_root:
            _d.dataset_root = str(Path(args.dataset_root).resolve())         # general configs name no dataset
        dt = OmegaConf.create({"data": _d}).data.train
    OmegaConf.set_struct(dt, False); dt.pretrained_norm_stats = None; dt.use_policy_code = False; dt.processor = None
    dt.policy_code_idx_path = None; dt.cosmos_text_cache_dir = None
    ds = instantiate(dt); lr = ds.lerobot_dataset
    frm = list(map(int, lr.episode_data_index["from"])); to = list(map(int, lr.episode_data_index["to"]))
    order = json.loads(Path(args.order).read_text())
    ok = len(order) == len(frm) and all(order[i]["offset"] == frm[i] and order[i]["length"] == to[i] - frm[i]
                                        for i in range(len(order)))
    print(f"episodes dataset={len(frm)} order={len(order)} total_frames={to[-1]} -> {'ALIGNED' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
