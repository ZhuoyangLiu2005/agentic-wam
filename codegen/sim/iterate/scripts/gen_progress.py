"""Regenerate the RoboTwin fix-loop progress table (the coordinator's routing input).

    pending      -> no complete Stage-1 output yet (fix_code.py without findings.md is TRUNCATED)
    stage1-done  -> fix_code.py exists (+ its debug score)
    done         -> Stage 2 has been run (locked seeds spent)

Regenerate before reading: the file goes stale as soon as a subagent finishes.

    python iterate/scripts/gen_progress.py [--tasks-file <file>] [--write]
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import tyro

_AGENT = Path(os.environ.get("CODEGEN_SIM_AGENT_ROOT", "outputs/agentic"))
FIX_ROOT = _AGENT / "fix_loop"
DEBUG_ROOT = _AGENT / "baseline"
STAGE2_ROOT = Path(os.environ.get("CODEGEN_SIM_VALIDATION_ROOT", "outputs/validation")) / "stage2"
STAGE2_EXPECTED = 15  # 5 clean (201-205) + 10 randomized (251-260)
STAGE2_LIVE_S = 900   # a stage2 dir written within 15 min is still running
STAGE1_LIVE_S = 1800  # a subagent between replays can be quiet for a while
PROGRESS_FILE = _AGENT / "progress" / "robotwin_fix_loop_progress.md"
_REWARD = re.compile(r"_reward_([0-9.]+)_")


@dataclass
class Args:
    tasks_file: str = "iterate/configs/tasks_all.txt"
    """One RoboTwin task name per line."""
    write: bool = True
    """Write the progress ledger under CODEGEN_SIM_AGENT_ROOT/progress (else print only)."""
    score_task: str | None = None
    """Print just this task's debug score as `<passed>/<total>` and exit.
    Other scripts gate on it, so it reuses the same layout-tolerant scan the
    table uses rather than reimplementing it."""


def _score(d: Path) -> tuple[int, int]:
    """(passed, total) over trial dirs directly under d."""
    if not d.is_dir():
        return (0, 0)
    trials = [p.name for p in d.iterdir() if p.name.startswith("trial_")]
    passed = sum(1 for t in trials if (m := _REWARD.search(t)) and float(m.group(1)) > 0)
    return (passed, len(trials))


def _baseline_score(task: str) -> tuple[int, int]:
    tot = pas = 0
    for cfgdir in DEBUG_ROOT.glob(f"*/{task}/*/run"):
        p, t = _score(cfgdir)
        pas += p
        tot += t
    return (pas, tot)


def _debug_score(task: str) -> tuple[int, int]:
    """Best per-seed outcome across this task's replay dirs (flat oracle/attempts_* or per-seed attempts_*/oracle/*)."""
    best: dict[str, float] = {}
    roots = list(DEBUG_ROOT.glob(f"oracle/*{task}*")) + list(DEBUG_ROOT.glob(f"*{task}*"))
    for d in roots:
        if not d.is_dir():
            continue
        for p in d.rglob("trial_*"):
            if not p.is_dir():
                continue
            seed = p.name.split("_")[1]
            m = _REWARD.search(p.name)
            if m:
                best[seed] = max(best.get(seed, 0.0), float(m.group(1)))
    return (sum(1 for v in best.values() if v > 0), len(best))


def _recently_written(d: Path, within_s: int) -> bool:
    """True if anything under d was written in the last within_s seconds."""
    newest = 0.0
    for f in d.rglob("*"):
        try:
            newest = max(newest, f.stat().st_mtime)
        except OSError:
            continue
    return newest > time.time() - within_s


def _stage1_live(task: str) -> bool:
    """True if the task's fix-loop dir or any debug replay dir was written recently."""
    dirs = [FIX_ROOT / task, *DEBUG_ROOT.glob(f"oracle/*{task}*")]
    return any(d.is_dir() and _recently_written(d, STAGE1_LIVE_S) for d in dirs)


def _findings_complete(f: Path) -> bool:
    """A findings.md is finished only once its LAST required section is present (killed writers leave partial files)."""
    if not f.exists():
        return False
    return "Collection suitability" in f.read_text(errors="ignore")


def state_of(task: str) -> dict:
    fix = FIX_ROOT / task / "fix_code.py"
    s2 = STAGE2_ROOT / task
    bp, bt = _baseline_score(task)
    dp, dt = _debug_score(task)
    if s2.is_dir() and any(s2.rglob("trial_*")):
        p = sum(_score(d)[0] for d in s2.glob("oracle/*"))
        t = sum(_score(d)[1] for d in s2.glob("oracle/*"))
        if t >= STAGE2_EXPECTED:
            st, note = "done", f"stage2 {p}/{t}"
        elif _recently_written(s2, STAGE2_LIVE_S):
            # Still being written: the run is alive; suggesting RESUME=1 would start a second concurrent run.
            st = "stage2-running"
            note = f"stage2 IN PROGRESS {p}/{t} of {STAGE2_EXPECTED} — do not start another"
        else:
            # Stale partial run: reporting `done` would publish a partial score and block finishing it.
            st = "stage2-partial"
            note = (f"STALLED stage2 {p}/{t} of {STAGE2_EXPECTED} — finish with "
                    f"`RESUME=1 scripts/run_stage2.sh {task}`")
    elif fix.exists() and _findings_complete(FIX_ROOT / task / "findings.md"):
        st, note = "stage1-done", f"debug {dp}/{dt}"
    elif fix.exists() and _stage1_live(task):
        # Mid-iteration subagent: re-dispatching would put a second subagent on the same fix_code.py.
        st, note = "stage1-running", f"subagent iterating (debug {dp}/{dt})"
    elif fix.exists():
        # Cut off mid-run: never stage1-done, or Stage 2 would spend locked seeds on it.
        st, note = "pending", f"TRUNCATED (fix_code but no findings; debug {dp}/{dt})"
    else:
        st, note = "pending", ""
    return {
        "task": task, "state": st, "baseline": f"{bp}/{bt}" if bt else "-",
        "debug": f"{dp}/{dt}" if dt else "-",
        "fix_code": fix.exists(), "note": note,
        "findings": _findings_complete(FIX_ROOT / task / "findings.md"),
    }


def main(args: Args) -> None:
    if args.score_task:
        p, t = _debug_score(args.score_task)
        print(f"{p}/{t}")
        return
    tasks = [
        t.strip() for t in Path(args.tasks_file).read_text().splitlines() if t.strip()
    ]
    rows = [state_of(t) for t in tasks]
    counts = {s: sum(1 for r in rows if r["state"] == s)
              for s in ("pending", "stage1-running", "stage1-done",
                        "stage2-running", "stage2-partial", "done")}

    lines = [
        "# RoboTwin fix-loop progress",
        "",
        "Regenerate with `python iterate/scripts/gen_progress.py` — never trust a stale copy.",
        "",
        f"**{counts['pending']} pending · {counts['stage1-running']} stage1-running · "
        f"{counts['stage1-done']} stage1-done · "
        f"{counts['stage2-running']} stage2-running · "
        f"{counts['stage2-partial']} stage2-partial · {counts['done']} done** "
        f"(of {len(rows)})",
        "",
        "| Task | State | Baseline | Debug (fix_code) | findings.md | Note |",
        "|---|---|---|---|---|---|",
    ]
    _ord = {"stage2-partial": 0, "stage1-done": 1,
            "stage2-running": 2, "stage1-running": 3}
    for r in sorted(rows, key=lambda r: (_ord.get(r["state"], 4), r["task"])):
        lines.append(
            f"| {r['task']} | {r['state']} | {r['baseline']} | {r['debug']} | "
            f"{'yes' if r['findings'] else 'no'} | {r['note']} |"
        )
    out = "\n".join(lines) + "\n"
    print(out)
    if args.write:
        PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROGRESS_FILE.write_text(out)
        (PROGRESS_FILE.with_suffix(".json")).write_text(json.dumps(rows, indent=1))
        print(f"-> {PROGRESS_FILE}")


if __name__ == "__main__":
    main(tyro.cli(Args))
