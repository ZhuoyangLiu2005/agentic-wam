#!/usr/bin/env python3
"""Promote a program into the handoff: frozen/<task>/{frozen_program.py, frozen_program.sha256, task.yaml}.

The pipeline side (collection, export, training, eval) reads nothing but this directory.

    iterate/scripts/freeze.py <task> --program <validated program.py> [--stage2 <STAGE2_RESULT.md>]
                              [--candidate <path> ...] [--embodiment aloha-agilex]
    iterate/scripts/freeze.py --refresh [<task> ...]      # rewrite sha256 + task.yaml for programs already in frozen/

task.yaml fields: task, program, sha256, stages, template_slots, candidates, record_instructions (static | expert |
off), embodiment, stage2 (score line of the STAGE2_RESULT.md that validated THIS sha, else null).
Stdlib + PyYAML only; runs under any python3 with the RoboTwin checkout resolvable (env.sh).
"""
import argparse
import hashlib
import importlib.util
import re
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "frozen"
_SCORE = re.compile(r"clean (\d+)/(\d+) \+ randomized (\d+)/(\d+) = (\d+)/(\d+)[^\n]*?\b(PASS|FAIL)\b")
_STAGE = re.compile(r"^\s*#\s*\[stage\s+(\d+)\s*/\s*(\d+)\]", re.M)
_EACH = re.compile(r"^\s*#\s*\[stage each\]", re.M)


def _placeholders():
    spec = importlib.util.spec_from_file_location("ph", ROOT / "capsim/integrations/robotwin/placeholders.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage2_of(result_md: Path | None, sha: str) -> dict | None:
    """The score line of a STAGE2_RESULT.md, only when its sibling frozen program (or quoted sha prefix) matches this sha."""
    if not result_md or not result_md.is_file():
        return None
    sibling = result_md.parent / "frozen_program.py"
    if sibling.is_file():
        if sha256(sibling) != sha:
            return None
    else:
        m_sha = re.search(r"sha256 `?([0-9a-f]{8,64})", result_md.read_text())
        if not m_sha or not sha.startswith(m_sha.group(1)):
            return None
    txt = result_md.read_text()
    m = _SCORE.search(txt)
    if not m:
        return None
    return {"clean": f"{m.group(1)}/{m.group(2)}", "randomized": f"{m.group(3)}/{m.group(4)}",
            "total": f"{m.group(5)}/{m.group(6)}", "verdict": m.group(7),
            "source": str(result_md.relative_to(ROOT)) if result_md.is_relative_to(ROOT) else str(result_md)}


def find_stage2(task: str, sha: str) -> dict | None:
    """A STAGE2_RESULT.md in frozen/<task>/ or among the iteration side's promotions whose sha matches."""
    for cand in [FROZEN / task / "STAGE2_RESULT.md", *sorted((ROOT / "iterate/local/promoted" / task).glob("**/STAGE2_RESULT.md"))]:
        s2 = stage2_of(cand, sha)
        if s2:
            return s2
    return None


def write_task_yaml(task: str, program_name: str, candidates: list[str], embodiment: str, stage2_md: Path | None) -> Path:
    d = FROZEN / task
    prog = d / program_name
    text = prog.read_text()
    sha = sha256(prog)
    (d / "frozen_program.sha256").write_text(f"{sha}  {program_name}\n")
    ph = _placeholders()
    n_stage = max((int(m.group(2)) for m in _STAGE.finditer(text)), default=0)
    if _EACH.search(text):
        n_stage = "each"                                # `[stage each]` loop: the instance count is per episode
    mode, _arms = ph.recommended_mode(task)
    meta = {
        "task": task,
        "program": program_name,
        "sha256": sha,
        "stages": n_stage,
        "template_slots": ph.template_slots(text),
        "candidates": sorted(candidates),
        "record_instructions": mode,
        "embodiment": embodiment,
        "stage2": stage2_of(stage2_md, sha) if stage2_md else find_stage2(task, sha),
    }
    out = d / "task.yaml"
    out.write_text("# Handoff record written by iterate/scripts/freeze.py - the pipeline reads this, never the iteration outputs.\n"
                   + yaml.safe_dump(meta, sort_keys=False, width=110))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="*")
    ap.add_argument("--program", help="validated program to promote (copied to frozen/<task>/frozen_program.py)")
    ap.add_argument("--stage2", help="its STAGE2_RESULT.md (copied next to it)")
    ap.add_argument("--candidate", action="append", default=[], help="another program to keep in the dir (not frozen)")
    ap.add_argument("--embodiment", default="aloha-agilex")
    ap.add_argument("--refresh", action="store_true", help="rewrite sha256 + task.yaml for existing frozen/<task> dirs")
    a = ap.parse_args()
    if a.refresh:
        tasks = a.task or sorted(p.name for p in FROZEN.iterdir() if (p / "frozen_program.py").is_file())
        for t in tasks:
            d = FROZEN / t
            cands = [p.name for p in d.glob("*.py") if p.name != "frozen_program.py"]
            out = write_task_yaml(t, "frozen_program.py", cands, a.embodiment, None)
            meta = yaml.safe_load(out.read_text())
            print(f"{t:26s} stages={meta['stages']} slots={meta['template_slots']} instr={meta['record_instructions']} "
                  f"stage2={(meta['stage2'] or {}).get('verdict')} cands={meta['candidates']}")
        return 0
    if len(a.task) != 1 or not a.program:
        sys.exit("usage: freeze.py <task> --program <program.py> [--stage2 <md>] | freeze.py --refresh [tasks]")
    task = a.task[0]
    d = FROZEN / task; d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.program, d / "frozen_program.py")
    if a.stage2:
        shutil.copyfile(a.stage2, d / "STAGE2_RESULT.md")
    cands = []
    for c in a.candidate:
        shutil.copyfile(c, d / Path(c).name); cands.append(Path(c).name)
    out = write_task_yaml(task, "frozen_program.py", cands, a.embodiment, d / "STAGE2_RESULT.md" if a.stage2 else None)
    print(f"promoted {a.program} -> {d}/frozen_program.py; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
