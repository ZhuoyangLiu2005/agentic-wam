#!/usr/bin/env python3
"""Evaluate the published RoboTwin checkpoints after the one-time setup in docs/ROBOTWIN.md.

    python scripts/eval_robotwin_release.py wam --tasks click_alarmclock --plan smoke --gpus 0
    python scripts/eval_robotwin_release.py hybrid --tasks click_alarmclock --plan smoke --gpus 0 --vis-gpu 1

Remaining arguments go to robotwin/eval/benchmark.py, including --dry-run,
--protocols, --episodes, --render and --launch-args (put --launch-args last).
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("wam", "hybrid"))
    parser.add_argument("--ckpt", help="override the released checkpoint")
    parser.add_argument("--vis-gpu", type=int, help="start/check hybrid perception on this GPU")
    parser.add_argument("--vis-gpu-seg", type=int, help="optional separate GPU for SAM3 and GraspNet")
    args, benchmark_args = parser.parse_known_args(argv)
    sys.path.insert(0, str(ROOT / "robotwin"))
    from hybrid_eval import config, vision

    model = "robotwin_" + args.mode
    try:
        site = config.site()
        preset = config.model_preset(model, site)
        checkpoint = config.resolve_checkpoint(preset, args.ckpt)
        required = [preset.get("dataset_stats") or site["dataset_stats"]]
        if args.mode == "wam":
            required.append(preset["env"]["AGENTICWAM_TASK_SKILL_PROMPT"])
        for path in required:
            if not Path(path).is_file():
                raise FileNotFoundError(f"release asset missing: {path}; download the complete model bundle")
    except (KeyError, ValueError, FileNotFoundError) as exc:
        parser.exit(1, str(exc) + "\n")

    if args.mode == "hybrid" and "--dry-run" not in benchmark_args:
        if args.vis_gpu is not None:
            if not vision.ensure_up(site, args.vis_gpu, args.vis_gpu_seg):
                return 1
        else:
            missing = [name for name, ready in vision.status(site).items() if not ready]
            if missing:
                parser.exit(1, "hybrid perception is not ready: " + ", ".join(missing)
                            + "; pass --vis-gpu N or start robotwin/hybrid_eval/vision.py first\n")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(ROOT / "robotwin/eval/benchmark.py"),
           "--model", model, "--mode", args.mode, "--ckpt", checkpoint, *benchmark_args]
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
