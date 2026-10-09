#!/usr/bin/env python3
"""One task, one command: frozen program -> aligned collection -> LeRobot dataset (+ slicing cache) -> dataset.yaml.

    source env.sh
    python pipeline/scripts/make_dataset.py --task click_alarmclock --episodes 60 --gpus 0 1 [--program <path>]
        [--start-seed 1001] [--task-config demo_clean] [--out $CODEGEN_SIM_DATA_ROOT/<task>] [--workers 8]
        [--skip-collect] [--skip-export] [--dry-run]

Output layout (<out>):
    collect.yaml                 the collection config actually used
    raw/                         data/*.hdf5, video/, alignment/*.json, manifest.jsonl, slices/ (stage cache, if the
                                 program carries stage markers — built by collect_aligned.sh)
    lerobot/robotwin2.0/         the LeRobot v2.1 dataset (capsim.data.lerobot_export)
    dataset.yaml                 manifest the downstream trainer consumes ($TRAINER_ROOT/robotwin/dataprep/prepare.py)

Without --program, uses `program` in frozen/<task>/task.yaml (written by iterate/scripts/freeze.py).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    return subprocess.run(["bash", "-c", f"source {ROOT}/env.sh && codegen_sim_preflight && {cmd}"], cwd=ROOT, **kw).returncode


def pick_program(task):
    """The task's frozen program: `program` in frozen/<task>/task.yaml (the handoff written by iterate/scripts/freeze.py)."""
    d = ROOT / "frozen" / task
    meta = d / "task.yaml"
    if not meta.is_file():
        sys.exit(f"no frozen handoff for {task}: {meta} (promote a program with iterate/scripts/freeze.py)")
    rel = yaml.safe_load(meta.read_text()).get("program") or "frozen_program.py"
    prog = d / rel
    if not prog.is_file():
        sys.exit(f"{meta} names a program that does not exist: {prog}")
    return prog


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--program", default=None)
    ap.add_argument("--episodes", type=int, default=60, help="seeds to run (successful ones are kept)")
    ap.add_argument("--start-seed", type=int, default=1001, help="collection seeds start at 1001 (clean) / 2001 (randomized)")
    ap.add_argument("--task-config", default="demo_clean")
    ap.add_argument("--gpus", nargs="+", default=["0"])
    ap.add_argument("--out", default=None, help="default $CODEGEN_SIM_DATA_ROOT/<task> (env.sh; outputs/data)")
    ap.add_argument("--workers", type=int, default=8, help="export parallelism")
    ap.add_argument("--n-instructions", type=int, default=10,
                    help="paraphrases per episode kept at export (first N of the recorded list; 0 = all). Each distinct one costs "
                         "~25 MB of trainer text cache")
    ap.add_argument("--instructions", choices=["auto", "static", "expert", "off"], default="auto",
                    help="placeholder source: auto (default) = static unless the task's rule names several arms, then expert; "
                         "static = the task's rule evaluated on the finished episode (no expert rollout); "
                         "expert = one expert rollout per scene; off = template only")
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    data_root = Path(os.environ.get("CODEGEN_SIM_DATA_ROOT") or ROOT / "outputs" / "data")
    out = Path(a.out or data_root / a.task).resolve()
    program = Path(a.program or pick_program(a.task)).resolve()
    if a.instructions == "auto":
        import importlib.util                    # load the pure-AST module by path: no simulator deps in the driver
        spec = importlib.util.spec_from_file_location("placeholders", ROOT / "capsim/integrations/robotwin/placeholders.py")
        ph = importlib.util.module_from_spec(spec); spec.loader.exec_module(ph)
        try:                                     # raises when the RoboTwin checkout is not found (env.sh not sourced):
            a.instructions, arms = ph.recommended_mode(a.task, os.environ.get("CAPX_ROBOTWIN_ROOT"))
        except FileNotFoundError as exc:         # never degrade to `off` silently - that ships a dataset without instructions
            sys.exit(f"cannot decide the instruction mode: {exc}")
        print(f"instructions: {a.instructions} (arm placeholder variables in {a.task}'s rule: {arms or 'none'})", flush=True)
    raw, lerobot = out / "raw", out / "lerobot" / "robotwin2.0"
    end_seed = a.start_seed + a.episodes - 1
    cfg = (ROOT / "pipeline/configs/collect_template.yaml").read_text()
    for k, v in {"__TASK__": a.task, "__TASK_CONFIG__": a.task_config, "__SAVE_DIR__": str(raw), "__PROGRAM__": str(program),
                 "__RUN_DIR__": str(out / "runs"), "__START_SEED__": str(a.start_seed), "__INSTRUCTIONS__": a.instructions}.items():
        cfg = cfg.replace(k, v)
    print(f"task={a.task} program={program.relative_to(ROOT) if program.is_relative_to(ROOT) else program} "
          f"seeds={a.start_seed}-{end_seed} ({a.task_config}) gpus={a.gpus} -> {out}", flush=True)
    if a.dry_run:
        print(cfg); return 0
    out.mkdir(parents=True, exist_ok=True)
    (out / "collect.yaml").write_text(cfg)

    if not a.skip_collect:
        rc = sh(f"pipeline/scripts/collect_aligned.sh {out / 'collect.yaml'} {a.start_seed} {end_seed} {' '.join(a.gpus)}")
        if rc:
            sys.exit(f"collection failed (rc={rc})")
    man = raw / "manifest.jsonl"
    rows = [json.loads(l) for l in open(man) if l.strip()] if man.exists() else []
    kept = sorted((r for r in rows if r.get("success")), key=lambda r: int(r["seed"]))
    n_seeds = end_seed - a.start_seed + 1
    print(f"seeds {a.start_seed}-{end_seed}: {len(rows)}/{n_seeds} produced an episode, {len(kept)} successful (kept); "
          f"{n_seeds - len(rows)} program failures left no episode", flush=True)
    if not kept:
        sys.exit("nothing to export")

    if not a.skip_export:
        embodiment = re.search(r"^\s*embodiment:\s*(\S+)", cfg, re.M).group(1)   # the collection's profile -> exporter
        rc = sh(': "${TRAIN_PY:?set train_py in site.yaml}" "${TRAINER_ROOT:?set trainer_root in site.yaml}" && '
                f'CAPSIM_EMBODIMENT={embodiment} "$TRAIN_PY" -m capsim.data.lerobot_export --raw-root {raw} --out {lerobot} --workers {a.workers}'
                + (f' --n-instructions {a.n_instructions}' if a.n_instructions else ''))
        if rc:
            sys.exit(f"export failed (rc={rc})")

    slices = raw / "slices"
    has_markers = bool(re.search(r"^\s*#\s*\[stage", program.read_text(), re.M))
    if has_markers and not (slices / "cycle_by_idx.pt").exists():      # collect_aligned.sh builds it; redo if missing
        rc = sh(f'"$TRAIN_PY" "$SLICER" build --program {program} --align-dir {raw / "alignment"} --out {slices}')
        if rc:
            print(f"WARNING: slicing cache not built (rc={rc}); dataset.yaml will carry slices: null", flush=True)
    has_slices = (slices / "cycle_by_idx.pt").exists()
    al0 = json.load(open(raw / "alignment" / f"episode{kept[0]['seed']}.json"))
    manifest = {
        "task": a.task, "task_config": a.task_config, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "program": str(program), "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
        "stage_markers": has_markers,
        "seeds": {"start": a.start_seed, "end": end_seed}, "episodes": {"run": len(rows), "kept": len(kept)},
        "raw_root": str(raw), "lerobot_dir": str(lerobot),
        "instructions": a.instructions, "instructions_per_episode": (lambda n: min(n, a.n_instructions) if a.n_instructions else n)(len((al0.get("instructions") or {}).get("seen") or [])),
        "records_have_top_line": bool(al0["records"] and "top_line" in al0["records"][0]),
        "slices": {k: str(slices / f) for k, f in (("preamble", "preamble.txt"), ("slices_ep", "slices_ep"), ("cycle_by_idx", "cycle_by_idx.pt"),
                    ("episodes_train", "episodes_train.json"), ("episodes_val", "episodes_val.json"),
                    ("episodes_all", "episodes_all.json"), ("stage_frames", "stage_frames.json"))} if has_slices else None,
    }
    (out / "dataset.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, width=200))
    print(f"wrote {out / 'dataset.yaml'}", flush=True)
    print("logs: the caller's stdout; long runs: redirect to $CODEGEN_SIM_LOG_ROOT/make_dataset_<task>.log", flush=True)
    print(f"next (downstream): python {os.environ.get('TRAINER_ROOT', '$TRAINER_ROOT')}/robotwin/dataprep/prepare.py "
          f"--datasets {out / 'dataset.yaml'} --out configs/robotwin/data/<name>.yaml", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
