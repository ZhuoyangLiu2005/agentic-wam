#!/usr/bin/env python3
"""Build the Cosmos-Reason1 prompt-embedding cache for codegen/sim datasets (required by the Cosmos backbone:
RobotVideoDataset reads <cache>/<sha256(prompt)>.cosmos_reason1.pt per frame instruction).

    python robotwin/dataprep/text_cache.py --datasets <d>/dataset.yaml ... --out-root DIR --ckpt <checkpoint dir> [--device cuda]

Instructions come from each dataset's LeRobot meta/tasks.jsonl (the filled paraphrases RoboTwin generated at
collection time); the cache for a task lands in <out-root>/<task> (~25 MB per distinct instruction — keep it on
node-local disk). Pass the same --out-root to data_config.py --text-cache-root.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ckpt", default=None, help="Cosmos-Reason1-7B directory (default: paths.text_encoder in configs/paths/default.yaml)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=128)
    a = ap.parse_args()
    if a.ckpt is None:
        sys.path.insert(0, str(REPO / 'src'))
        from agenticwam.utils.paths import default_path
        a.ckpt = default_path('text_encoder')
    for p in a.datasets:
        m = yaml.safe_load(open(p))
        tasks = Path(m["lerobot_dir"]) / "meta" / "tasks.jsonl"
        instr = [json.loads(l)["task"] for l in open(tasks) if l.strip()]
        out = Path(a.out_root) / m["task"]; out.mkdir(parents=True, exist_ok=True)
        lst = out / "_instructions.txt"; lst.write_text("\n".join(instr) + "\n")
        print(f"[{m['task']}] {len(instr)} distinct instructions -> {out}", flush=True)
        rc = subprocess.run([sys.executable, str(REPO / "scripts" / "precompute_cosmos_text_embeds.py"), "--instructions", str(lst),
                             "--out-dir", str(out), "--ckpt", a.ckpt, "--max-len", str(a.max_len), "--device", a.device]).returncode
        if rc:
            sys.exit(f"{m['task']}: text cache build failed (rc={rc})")
    print("ok")


if __name__ == "__main__":
    main()
