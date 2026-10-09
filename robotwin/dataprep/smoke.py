#!/usr/bin/env python3
"""Instantiate the training dataset of a data config and fetch a few samples — the proof that a codegen/sim
dataset is consumable here. Run with the trainer env:

    python robotwin/dataprep/smoke.py configs/robotwin/data/<name>.yaml [--n 2]
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config"); ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(a.config)
    ds = instantiate(cfg.train.dataset)          # ConcatDataset of RobotVideoDataset
    print(f"datasets: {len(ds.datasets)} | windows: {len(ds)}")
    for i in range(min(a.n, len(ds))):
        s = ds[i * (len(ds) // max(a.n, 1))]
        shapes = {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in s.items()}
        print(f"sample {i}: {shapes}")
        for k in ("policy_code", "prompt_code", "instruction", "task"):
            if k in s and isinstance(s[k], str):
                print(f"  {k}: {s[k].splitlines()[0][:100]!r} ({len(s[k])} chars)")
    print("ok")


if __name__ == "__main__":
    main()
