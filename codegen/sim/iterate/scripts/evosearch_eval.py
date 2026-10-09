"""Evaluate K evolutionary-search candidates (``candidate_*.py`` in an iteration dir) on the debug seeds.

Writes <iter_dir>/candidate_X/eval/trial_*/ per seed and <iter_dir>/iter_summary.json (the leaderboard).
Debug seeds only (101-105, 151-155); the locked validation set is spent once by ``run_stage2.sh``.

    $PY iterate/scripts/evosearch_eval.py \
        --iter-dir $CODEGEN_SIM_AGENT_ROOT/evosearch/<task>/<run>/<iteration> \
        --task <task> --gpus <assigned_gpu_ids> [--resume]   # --resume skips seeds with a trial dir
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import tyro

SIM_ROOT = Path(__file__).resolve().parents[2]
PY = os.environ.get("ROBOTWIN_PY", "python")   # env.sh
CFG_CLEAN = "iterate/configs/priv_clean.yaml"
CFG_RAND = "iterate/configs/priv_randomized.yaml"
_REWARD = re.compile(r"_reward_([0-9.]+)_")

# The locked validation range: tuning against it invalidates the reported number.
LOCKED = set(range(201, 211)) | set(range(251, 271))


@dataclass
class Args:
    iter_dir: str
    """Directory holding candidate_*.py for this iteration."""

    task: str
    """RoboTwin task name."""

    gpus: list[int] = field(default_factory=lambda: [0])
    """GPU indices to spread replays across."""

    clean_seeds: list[int] = field(default_factory=lambda: [101, 102, 103, 104, 105])
    """Debug seeds run against the clean scene config."""

    rand_seeds: list[int] = field(default_factory=lambda: [151, 152, 153, 154, 155])
    """Debug seeds run against the randomized scene config."""

    per_gpu: int = 2
    """Concurrent replays per GPU (~6 GB each)."""

    record_video: bool = False
    """Videos are large; keep them off during search and on for the final pick."""

    resume: bool = False
    """Skip (candidate, seed) pairs that already have a trial directory."""


def _trial_dirs(root: Path) -> list[Path]:
    """Every trial dir under root, at any depth (``launch`` inserts the model name as an extra level)."""
    return [p for p in root.rglob("trial_*") if p.is_dir()] if root.is_dir() else []


def _seed_of(p: Path) -> str:
    return p.name.split("_")[1]


def _score(cand_root: Path) -> dict[str, float]:
    """Best reward per seed for one candidate."""
    best: dict[str, float] = {}
    for p in _trial_dirs(cand_root):
        m = _REWARD.search(p.name)
        if m:
            s = _seed_of(p)
            best[s] = max(best.get(s, 0.0), float(m.group(1)))
    return best


def main(args: Args) -> None:
    os.chdir(SIM_ROOT)
    iter_dir = Path(args.iter_dir)
    if not iter_dir.is_dir():
        sys.exit(f"no such iteration dir: {iter_dir}")

    bad = (set(args.clean_seeds) | set(args.rand_seeds)) & LOCKED
    if bad:
        sys.exit(
            f"REFUSING: seeds {sorted(bad)} are the locked validation set. "
            "Iterate on 101-105 / 151-155 only; run_stage2.sh spends the locked "
            "seeds once, after the search converges."
        )

    candidates = sorted(p for p in iter_dir.glob("candidate_*.py"))
    if not candidates:
        sys.exit(f"no candidate_*.py in {iter_dir}")

    jobs = []
    for cand in candidates:
        out = iter_dir / cand.stem / "eval"
        (out / "logs").mkdir(parents=True, exist_ok=True)
        for seeds, cfg, tag in ((args.clean_seeds, CFG_CLEAN, "clean"),
                                (args.rand_seeds, CFG_RAND, "randomized")):
            done_seeds = {_seed_of(p) for p in _trial_dirs(iter_dir / cand.stem)}
            for seed in seeds:
                if args.resume and str(seed) in done_seeds:
                    continue
                jobs.append((cand, out, seed, cfg, tag))

    if not jobs:
        print("nothing to run — every (candidate, seed) already has a trial dir")
    else:
        gpus = args.gpus or [0]
        conc = max(1, len(gpus) * args.per_gpu)
        print(f"evosearch_eval: {len(candidates)} candidates × "
              f"{len(args.clean_seeds) + len(args.rand_seeds)} seeds = {len(jobs)} replays "
              f"| gpus={gpus} concurrency={conc}")

        def run(i_job):
            i, (cand, out, seed, cfg, tag) = i_job
            gpu = gpus[i % len(gpus)]
            log = out / "logs" / f"{tag}_{seed}.log"
            cmd = [PY, "iterate/scripts/replay_trial.py", "--code", str(cand),
                   "--seed", str(seed), "--task", args.task, "--config", cfg,
                   "--output-dir", str(out)]
            cmd += ["--record-video"] if args.record_video else ["--no-record-video"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            with open(log, "w", encoding="utf-8") as fh:
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
            print(f"  {cand.stem} seed {seed} ({tag}, gpu {gpu}) rc={rc}", flush=True)

        with ThreadPoolExecutor(max_workers=conc) as pool:
            list(pool.map(run, enumerate(jobs)))

    clean_set = {str(s) for s in args.clean_seeds}
    rows = []
    for cand in candidates:
        best = _score(iter_dir / cand.stem)
        passes = sum(1 for v in best.values() if v > 0)
        clean = sum(1 for s, v in best.items() if s in clean_set and v > 0)
        rand = passes - clean
        logs = iter_dir / cand.stem / "eval" / "logs"
        errors = sum(
            1 for p in logs.glob("*.log") if "Traceback" in p.read_text(errors="ignore")
        ) if logs.is_dir() else 0
        # Requested seeds with no trial dir are failures, not absences.
        missing = sorted(
            str(s) for s in (*args.clean_seeds, *args.rand_seeds) if str(s) not in best
        )
        trials = len(args.clean_seeds) + len(args.rand_seeds)
        rows.append({
            "candidate": cand.stem, "pass_count": passes, "trials": trials,
            "pass_rate": (passes / trials) if trials else 0.0,
            "clean": clean, "randomized": rand, "errors": errors,
            "failed_seeds": sorted(s for s, v in best.items() if v <= 0),
            "missing_seeds": missing,
        })

    rows.sort(key=lambda r: -r["pass_rate"])
    summary = {
        "iter_dir": str(iter_dir), "task": args.task,
        "seeds": {"clean": args.clean_seeds, "randomized": args.rand_seeds},
        "candidates": rows,
        "best_candidate": rows[0]["candidate"] if rows else None,
        "best_pass_rate": rows[0]["pass_rate"] if rows else 0.0,
    }
    (iter_dir / "iter_summary.json").write_text(json.dumps(summary, indent=1))

    nc, nr = len(args.clean_seeds), len(args.rand_seeds)
    print(f"\n=== {iter_dir} ===")
    print(f"{'candidate':<16}{'rate':>7}  {'clean':>6} {'rand':>7} {'err':>4}  failed / [missing]")
    for r in rows:
        miss = f"  [missing {' '.join(r['missing_seeds'])}]" if r["missing_seeds"] else ""
        print(f"{r['candidate']:<16}{r['pass_rate']:>6.0%}  {r['clean']:>3}/{nc:<2} "
              f"{r['randomized']:>4}/{nr:<2} {r['errors']:>4}  {' '.join(r['failed_seeds'])}{miss}")
    print(f"\nbest: {summary['best_candidate']} ({summary['best_pass_rate']:.0%})")
    print(f"-> {iter_dir / 'iter_summary.json'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
