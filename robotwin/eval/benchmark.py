#!/usr/bin/env python3
"""Evaluate one checkpoint on the RoboTwin 2.0 benchmark: every task of a suite, in both settings, sharded
over the GPUs of this node, resumable, then a results table.

    python robotwin/eval/benchmark.py --model robotwin_wam --ckpt <step.pt> \\
        [--suite robotwin] [--protocols clean,randomized] [--plan standard | smoke] \\
        [--tasks a,b,...] [--gpus 0,1,2,3,4,5,6,7] [--per-gpu 1] [--mode wam | hybrid] [--name NAME] \\
        [--episodes N --shards S] [--retries 1] [--render] [--dry-run]

What runs:
  * suites / protocols / plans come from configs/robotwin/benchmark.yaml; the checkpoint family's settings (eval
    task config, action space, decode, env) from a preset in configs/robotwin/models/.
  * one job = (task, protocol, shard). Shard k evaluates ceil(N/S) episodes with RoboTwin seed seed_base + k, through
    robotwin/hybrid_eval/launch.py (which writes run_meta.json with the result beside the rollouts).
  * jobs are scheduled on --gpus (--per-gpu at a time on each); a finished job (run_meta with a result) is skipped,
    so rerunning the same command resumes; a job that fails is retried --retries times.
  * the manifest, per-job logs and the report go to evaluate_results/robotwin_benchmark/<name>/.

Hybrid mode (generated code x WAM) needs a preset with task assets and the vision servers already running
(robotwin/hybrid_eval/vision.py start); its runs are also browsable with robotwin/hybrid_eval/rollout_viewer.py.
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROBOTWIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOTWIN))
from hybrid_eval import config  # noqa: E402
from hybrid_eval.launch import ckpt_tag  # noqa: E402

import yaml  # noqa: E402

REPO = config.REPO
LAUNCH = ROBOTWIN / "hybrid_eval" / "launch.py"
BENCH_ROOT = REPO / "evaluate_results" / "robotwin_benchmark"


def load_benchmark():
    return yaml.safe_load((config.CFG_DIR / "benchmark.yaml").read_text(encoding="utf-8"))


def results_dir(ckpt, tag, task, seed):
    """Where launch.py puts a job's results (mirror of launch.plan)."""
    return REPO / "evaluate_results" / "robotwin" / ckpt_tag(ckpt) / f"{tag}_{task}_g0_s{seed}" / task


def job_result(job):
    """(success, total) of a finished job, else None."""
    p = Path(job["results_dir"]) / "run_meta.json"
    try:
        r = (json.loads(p.read_text()).get("result") or {})
    except (OSError, ValueError):
        return None
    if r.get("rc", 0) != 0:
        return None
    if "success" in r and r.get("total"):
        success, total = int(r["success"]), int(r["total"])
        if total == int(job.get("episodes", total)):
            return success, total
    return None


def build_jobs(a, bench):
    suite = a.tasks.split(",") if a.tasks else bench["suites"][a.suite]
    unknown = [t for t in suite if t not in bench["suites"]["robotwin"]]
    if unknown:
        sys.exit(f"unknown RoboTwin task(s): {unknown}")
    plan = bench["plans"][a.plan]
    jobs = []
    for proto in a.protocols.split(","):
        if proto not in bench["protocols"]:
            sys.exit(f"unknown protocol {proto!r} (known: {', '.join(bench['protocols'])})")
        p = dict(plan[proto])
        n = a.episodes or p["episodes"]
        s = a.shards or p["shards"]
        per = math.ceil(n / s)
        tag = f"{a.name}_{proto}"
        for task in suite:
            left = n
            for k in range(s):
                if left <= 0:
                    break
                eps = min(per, left)
                left -= eps
                seed = p["seed_base"] + k
                jobs.append({"task": task, "protocol": proto, "task_config": bench["protocols"][proto]["task_config"],
                             "shard": k, "seed": seed, "episodes": eps, "tag": tag,
                             "results_dir": str(results_dir(a.ckpt, tag, task, seed))})
    return jobs


def command(a, job, gpu):
    cmd = [sys.executable, str(LAUNCH), "--mode", a.mode, "--model", a.model, "--ckpt", a.ckpt,
           "--task", job["task"], "--proto", job["task_config"], "--seed", str(job["seed"]),
           "--episodes", str(job["episodes"]), "--shard-gpu", str(gpu), "--tag", job["tag"]]
    if not a.render:
        cmd.append("--no-render")
    if a.mode == "hybrid":
        cmd.append("--no-vision")
    return cmd + a.launch_args


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="preset in configs/robotwin/models/ (or a yaml path)")
    ap.add_argument("--ckpt", default=None, help="checkpoint .pt (default: model preset's released checkpoint)")
    ap.add_argument("--suite", default="robotwin", help="task suite in configs/robotwin/benchmark.yaml")
    ap.add_argument("--tasks", default=None, help="comma list; overrides --suite")
    ap.add_argument("--protocols", default="clean,randomized")
    ap.add_argument("--plan", default="standard", help="plans in benchmark.yaml: standard | smoke")
    ap.add_argument("--episodes", type=int, default=None, help="per task and protocol (overrides the plan)")
    ap.add_argument("--shards", type=int, default=None, help="processes per task and protocol (overrides the plan)")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--per-gpu", type=int, default=1, help="concurrent jobs per GPU")
    ap.add_argument("--mode", choices=["wam", "hybrid"], default="wam")
    ap.add_argument("--name", default=None, help="default: <model>_<checkpoint stem>")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--render", action="store_true", help="write rollout videos (off by default: disk)")
    ap.add_argument("--dry-run", action="store_true", help="print the job plan and the first commands")
    ap.add_argument("--launch-args", nargs=argparse.REMAINDER, default=[],
                    help="everything after this is passed to launch.py verbatim")
    a = ap.parse_args()
    try:
        a.ckpt = config.resolve_checkpoint(config.model_preset(a.model), a.ckpt)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(str(exc))
    a.name = a.name or f"{Path(a.model).stem}_{Path(a.ckpt).stem}"
    bench = load_benchmark()
    jobs = build_jobs(a, bench)
    out = BENCH_ROOT / a.name
    manifest = {"name": a.name, "model": a.model, "ckpt": a.ckpt, "mode": a.mode, "suite": a.suite,
                "tasks": sorted({j["task"] for j in jobs}), "protocols": a.protocols.split(","), "plan": a.plan,
                "created": datetime.now().isoformat(timespec="seconds"), "jobs": jobs}
    old = out / "manifest.json"
    if old.is_file() and json.loads(old.read_text()).get("ckpt") != a.ckpt:
        sys.exit(f"{old} belongs to another checkpoint; pick another --name")
    if not a.dry_run:
        (out / "logs").mkdir(parents=True, exist_ok=True)
        old.write_text(json.dumps(manifest, indent=1))

    todo = [j for j in jobs if job_result(j) is None]
    n_eps = sum(j["episodes"] for j in jobs)
    print(f"[benchmark] {a.name}: {len(jobs)} jobs ({len(manifest['tasks'])} tasks x {a.protocols} x shards), "
          f"{n_eps} episodes; {len(jobs) - len(todo)} already done, {len(todo)} to run on GPUs {a.gpus} x{a.per_gpu}")
    if a.dry_run:
        for j in todo[:3]:
            print("  " + " ".join(command(a, j, a.gpus.split(",")[0])))
        return 0

    slots = [g for g in a.gpus.split(",") for _ in range(a.per_gpu)]
    queue = [dict(j, attempts=0) for j in todo]
    running = {}                                   # slot index -> (job, Popen, log handle)
    failed = []
    t0 = time.time()

    def stop(*_):
        for _, p, _ in running.values():
            p.terminate()
        sys.exit("[benchmark] interrupted; rerun the same command to resume")
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    done = len(jobs) - len(todo)
    while queue or running:
        for i, g in enumerate(slots):
            if i not in running and queue:
                j = queue.pop(0)
                j["attempts"] += 1
                log = open(out / "logs" / f"{j['protocol']}_{j['task']}_s{j['seed']}.log", "a")
                running[i] = (j, subprocess.Popen(command(a, j, g), cwd=REPO, stdout=log, stderr=subprocess.STDOUT), log)
        time.sleep(10)
        for i in list(running):
            j, p, log = running[i]
            if p.poll() is None:
                continue
            log.close()
            del running[i]
            if job_result(j) is not None:
                done += 1
            elif j["attempts"] <= a.retries:
                queue.append(j)
            else:
                failed.append(j)
            print(f"[benchmark] {datetime.now():%H:%M} {done}/{len(jobs)} done, {len(running)} running, "
                  f"{len(queue)} queued, {len(failed)} failed  ({(time.time() - t0) / 3600:.1f} h)", flush=True)

    for j in failed:
        print(f"[benchmark] FAILED {j['protocol']} {j['task']} seed {j['seed']}: see {out / 'logs'}")
    subprocess.run([sys.executable, str(Path(__file__).with_name("report.py")), a.name], cwd=REPO)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
