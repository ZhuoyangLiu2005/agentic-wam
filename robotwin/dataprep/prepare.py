#!/usr/bin/env python3
"""From codegen/sim dataset manifests to a launchable training run, in one command.

    python robotwin/dataprep/prepare.py --datasets <sim>/outputs/data/*/dataset.yaml --name my_tasks \
        [--norm-stats PATH] [--text-cache-root DIR] [--text-encoder DIR] [--foundation-ckpt PATH] [--code-vision-path DIR]
        [--split train|all] [--skip-text-cache] [--skip-smoke] [--no-task-config]

Steps (each is also a standalone script in this directory):
  1. text_cache.py   Cosmos-Reason1 prompt embeddings for every instruction in the datasets -> <text-cache-root>/<task>/
  2. data_config.py  configs/robotwin/data/<name>.yaml   (one RobotVideoDataset block per manifest on data_base.yaml)
  3. task_template   configs/robotwin/task/<name>.yaml   (override /data: <name>; Cosmos / SDAR / foundation / ViT-prefix paths filled in,
                     so configs/model/agenticwam_cosmos.yaml needs no local edits)
  4. smoke.py        instantiate the data config and fetch samples

Machine paths come from flags, else from `dataprep:` in configs/robotwin/site.yaml (see site.example.yaml).
Run with the trainer env. Afterwards:  bash scripts/train_robotwin.sh <name>
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

SITE_KEYS = ("norm_stats", "text_cache_root", "text_encoder", "code_model", "foundation_ckpt", "code_vision_path")


def site_defaults():
    """`dataprep:` block of site.yaml, resolved; {} when there is no site file."""
    try:
        from hybrid_eval import config
        s = config.site()
        return dict(s.get("dataprep") or {})
    except Exception:
        return {}


VIT_PREFIX_TOKENS = 210   # qwen3vl_prefix tokens that share the code expert's RoPE window (code_vision in task_template)


def code_token_budget(manifests, floor=1280):
    """max_code_tokens for the task config: the longest rendered code prompt (`# Instruction`, helper library,
    `# Next:`, slice) over every episode slice of every dataset, plus the ViT prefix and headroom, rounded up to 128."""
    import json, math, yaml
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REPO / "checkpoints" / "SDAR-1.7B-Chat"))
    n = lambda t: len(tok.encode(t, add_special_tokens=False))
    worst = 0
    for m in manifests:
        m = yaml.safe_load(open(m))
        if not m.get("slices"):
            continue
        pre = Path(m["slices"]["preamble"]).read_text()
        head = n(f"# Instruction: {'x' * 120}\n# Helper library:\n{pre}# Next:\n")
        for f in sorted(Path(m["slices"]["slices_ep"]).glob("ep_*.json")):
            for sl in json.load(open(f)).get("slices", []):
                worst = max(worst, head + n(sl))
    return max(floor, int(math.ceil((worst + VIT_PREFIX_TOKENS + 64) / 128) * 128))


def run(argv):
    print("+", " ".join(str(a) for a in argv), flush=True)
    if subprocess.run([sys.executable] + [str(a) for a in argv]).returncode:
        sys.exit(f"failed: {argv[0]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True, help="dataset.yaml manifests from codegen/sim")
    ap.add_argument("--name", required=True, help="stem of configs/robotwin/{data,task}/<name>.yaml")
    ap.add_argument("--split", choices=["train", "all"], default="train")
    for k in SITE_KEYS:
        ap.add_argument("--" + k.replace("_", "-"), default=None, help=f"default: site.yaml dataprep.{k}")
    ap.add_argument("--skip-text-cache", action="store_true", help="the cache already exists")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--no-task-config", action="store_true", help="only the data config")
    a = ap.parse_args()

    site = site_defaults()
    v = {k: getattr(a, k) or site.get(k) for k in SITE_KEYS}
    missing = [k for k in ("norm_stats",) if not v[k]]
    if not a.skip_text_cache and not v["text_cache_root"]:
        missing.append("text_cache_root")
    if not a.no_task_config:
        missing += [k for k in ("text_encoder", "code_model", "foundation_ckpt", "code_vision_path") if not v[k]]
    if missing:
        sys.exit("missing: " + ", ".join(f"--{k.replace('_', '-')} (or site.yaml dataprep.{k})" for k in missing))

    if not a.skip_text_cache:
        cmd = [HERE / "text_cache.py", "--datasets", *a.datasets, "--out-root", v["text_cache_root"]]
        if v["text_encoder"]:
            cmd += ["--ckpt", v["text_encoder"]]
        run(cmd)

    data_cfg = REPO / "configs" / "robotwin" / "data" / f"{a.name}.yaml"
    cmd = [HERE / "data_config.py", "--datasets", *a.datasets, "--out", data_cfg, "--split", a.split,
           "--norm-stats", v["norm_stats"]]
    if v["text_cache_root"]:
        cmd += ["--text-cache-root", v["text_cache_root"]]
    run(cmd)

    if not a.no_task_config:
        task_cfg = REPO / "configs" / "robotwin" / "task" / f"{a.name}.yaml"
        budget = code_token_budget(a.datasets)
        text = ((HERE / "task_template.yaml").read_text()
                .replace("__DATA__", a.name)
                .replace("__MAX_CODE_TOKENS__", str(budget))
                .replace("__CODE_MODEL__", str(v["code_model"]))
                .replace("__FOUNDATION_CKPT__", str(v["foundation_ckpt"]))
                .replace("__CODE_VISION_PATH__", str(v["code_vision_path"])))
        task_cfg.write_text(text)
        print(f"wrote {task_cfg} (max_code_tokens={budget})")

    if not a.skip_smoke:
        run([HERE / "smoke.py", data_cfg])

    print(f"\nready: bash scripts/train_robotwin.sh {a.name} [batch_size=4 max_steps=15000]")


if __name__ == "__main__":
    main()
