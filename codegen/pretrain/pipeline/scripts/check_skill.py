#!/usr/bin/env python
"""Admission checker for a candidate composed skill — the coordinator runs this
BEFORE leaving a new skill_library/skills/<name>/ in place.

Because nothing executes in the inverse pipeline, a skill's "replay" is free
static analysis. Three checks, all must pass:

  1. LOADER VALIDATION — codegen.library.validate_skill_source: exactly the
     gate the actors apply at startup (straight-line body, robot.<core-api>
     calls only, legal literal sides). An invalid skill.py is silently DROPPED
     by actors, so a failing candidate must never be left in place.
  2. CONTRACT DERIVATION — the skill's ordered gripper contract, derived by the
     SAME expansion machinery the grounder uses (call-site arg binding through
     the API contract table). Side parameters appear symbolically, e.g.
     transfer_grip -> close(<to_side>), open(<from_side>).
  3. EVIDENCE CHECK (--evidence, repeatable) — for each cited slice
     <task>:<run_dir_name>:<slice_idx>, some assignment of the skill's side
     parameters to left/right must make the contract a CONTIGUOUS subsequence
     of that slice's recorded observed gripper events. This is the pin-replay
     admission gate: the skill's contract must be attested by the runs cited
     in its SKILL.md.

With --vanilla the third check is replaced by the ANNOTATION-ERA evidence
check (library law, VANILLA bar): each cited slice must come from a vanilla
run and contain the skill's call sequence in its CODE; observed detector
events are not consulted, and the slice's annotation label is printed for
the log entry.

The skill library, contract table, and evidence paths are those of the ACTIVE
embodiment (CODEGEN_EMBODIMENT env, default yam). Usage (workspace root):
    $PY scripts/check_skill.py <name> \\
        --evidence <dataset>:<model>/episode_000/<tag>:<slice>
    CODEGEN_EMBODIMENT=ego $PY scripts/check_skill.py <name> \\
        --evidence <dataset>:<model>/episode_000/<tag>:<slice>
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codegen.api.active import _ARM_PARAM, EMBODIMENT, gripper_effects  # noqa: E402
from codegen.library import validate_skill_source  # noqa: E402
from codegen.skill_paths import library_dir  # noqa: E402
from codegen.program.calls import _bound_args  # noqa: E402

SIDES = ("left", "right")


def find_skill(name: str) -> Path | None:
    for base in ("skills", "_attic"):
        p = library_dir(EMBODIMENT) / base / name / "skill.py"
        if p.exists():
            return p
    return None


def _robot_calls(src: str) -> list[ast.Call]:
    """Ordered robot.<fn>(...) calls of the skill body."""
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef))
    return sorted(
        (n for n in ast.walk(fn) if isinstance(n, ast.Call)
         and isinstance(n.func, ast.Attribute)
         and isinstance(n.func.value, ast.Name) and n.func.value.id == "robot"),
        key=lambda n: (n.lineno, n.col_offset))


def derive_contract(src: str) -> list[tuple[str, str]]:
    """Ordered [(arm, kind)] of the skill body, exactly as the grounder derives
    it at a call site: formals bind symbolically as '<formal>'; hardcoded
    'left'/'right' literals pass through."""
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef))
    env = {a.arg: f"<{a.arg}>" for a in fn.args.args[1:]}   # skip `robot`
    calls = _robot_calls(src)
    seq = []
    for call in calls:
        name = call.func.attr
        bound = _bound_args(name, call, env)
        for ref, kind in (gripper_effects(name, bound) or []):
            arm = ref if ref in SIDES else bound.get(_ARM_PARAM[ref])
            if not isinstance(arm, str):
                raise ValueError(
                    f"line {call.lineno}: robot.{name} — its side argument is neither a "
                    f"literal nor a skill parameter; the contract is underivable")
            seq.append((arm, kind))
    return seq


def contract_str(seq: list[tuple[str, str]]) -> str:
    return " -> ".join(f"{k}({a})" for (a, k) in seq) or "(gripper-transparent)"


def check_evidence(seq: list[tuple[str, str]], ref: str) -> tuple[bool, str]:
    try:
        task, run_dir, idx = ref.rsplit(":", 2)
        slice_idx = int(idx)
    except ValueError:
        return False, f"bad --evidence ref {ref!r} (want <task>:<run_dir_name>:<slice_idx>)"
    base = ROOT / "outputs" / EMBODIMENT
    tr = base / task / run_dir / "trace.json"
    if not tr.exists():
        return False, f"{tr.relative_to(ROOT)} not found"
    slices = json.loads(tr.read_text()).get("slices", [])
    if not (0 <= slice_idx < len(slices)):
        return False, f"slice {slice_idx} out of range (run has {len(slices)})"
    gr = slices[slice_idx].get("grip_reconcile") or {}
    observed = gr.get("observed_grippers") or []
    if not seq:
        return True, f"contract is empty (transparent) — trivially consistent; " \
                     f"cite gripper evidence only for skills WITH a contract"
    markers = sorted({a for (a, _k) in seq if a not in SIDES})
    for assign in itertools.product(SIDES, repeat=len(markers)):
        if len(set(assign)) != len(assign):
            continue   # DISTINCT side params denote DISTINCT arms — a same-arm
                       # assignment would let e.g. a plain pick attest a transfer
        m = dict(zip(markers, assign))
        toks = [f"{k}_{m.get(a, a)}" for (a, k) in seq]
        for i in range(len(observed) - len(toks) + 1):
            if observed[i:i + len(toks)] == toks:
                binding = ", ".join(f"{k}={v}" for k, v in m.items()) or "(no side params)"
                return True, f"matched at observed[{i}:{i + len(toks)}] with {binding} " \
                             f"(slice matched={gr.get('matched')})"
    return False, f"contract not a contiguous subsequence of observed={observed}"


def check_vanilla_evidence(body_calls: list[str], ref: str) -> tuple[bool, str]:
    """Annotation-chain evidence (skill_library <emb>/CLAUDE.md): the cited
    slice must come from a run whose actor prompt carried NO detector events
    (prompt_events=false: pure vanilla, or the ego annotation-only default)
    and its CODE must contain the skill's robot-call sequence in order (gaps
    allowed). Observed detector events play no part; the slice's annotation
    label is reported so the coordinator can log whether it names the
    maneuver."""
    import re
    try:
        task, run_dir, idx = ref.rsplit(":", 2)
        slice_idx = int(idx)
    except ValueError:
        return False, f"bad --evidence ref {ref!r} (want <task>:<run_dir_name>:<slice_idx>)"
    tr = ROOT / "outputs" / EMBODIMENT / task / run_dir / "trace.json"
    if not tr.exists():
        return False, f"{tr.relative_to(ROOT)} not found"
    data = json.loads(tr.read_text())
    # Admissible evidence = any run whose ACTOR PROMPT carried no detector
    # events (pure vanilla, or the ego default annotation-only mode) — the
    # code is then detector-independent. Old traces lack prompt_events; infer
    # it from the vanilla flag (old standard runs had events in the prompt).
    if data.get("prompt_events", not data.get("vanilla", False)):
        return False, ("run's actor prompt CONTAINED detector events — annotation-chain "
                       "admissions cite prompt_events=false runs only")
    slices = data.get("slices", [])
    if not (0 <= slice_idx < len(slices)):
        return False, f"slice {slice_idx} out of range (run has {len(slices)})"
    sl = slices[slice_idx]
    called = re.findall(r"robot\.(\w+)\(", "\n".join(sl.get("code", {}).get("lines") or []))
    it = iter(called)
    if not all(c in it for c in body_calls):        # in-order subsequence
        return False, f"skill call sequence {body_calls} not found in slice code calls {called}"
    return True, f"call sequence present in slice code; annotation label: {sl.get('label')!r}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="skill dir name under skill_library/skills/ (or _attic/)")
    ap.add_argument("--evidence", action="append", default=[],
                    help="<task>:<run_dir_name>:<slice_idx> — a slice whose observed "
                         "events must contain the skill's contract (repeatable)")
    ap.add_argument("--vanilla", action="store_true",
                    help="VANILLA admission chain (library law, VANILLA bar): "
                         "evidence slices must come from vanilla runs and CONTAIN "
                         "the skill's call sequence in their code; observed "
                         "detector events are NOT consulted")
    args = ap.parse_args()

    py = find_skill(args.name)
    if py is None:
        print(f"FAIL: no skill_library/{{skills,_attic}}/{args.name}/skill.py")
        return 1
    src = py.read_text()
    print(f"skill: {py.relative_to(ROOT)}")

    probs = validate_skill_source(src, name=args.name)
    if probs:
        print("FAIL loader validation (actors would silently drop this skill):")
        for p in probs:
            print(f"  - {p}")
        return 1
    print("PASS loader validation")

    try:
        seq = derive_contract(src)
    except ValueError as e:
        print(f"FAIL contract derivation: {e}")
        return 1
    print(f"PASS contract derivation: {contract_str(seq)}")

    body_calls = [c.func.attr for c in _robot_calls(src)]
    bad = 0
    for ref in args.evidence:
        ok, msg = (check_vanilla_evidence(body_calls, ref) if args.vanilla
                   else check_evidence(seq, ref))
        print(f"{'PASS' if ok else 'FAIL'} evidence {ref}: {msg}")
        bad += (not ok)
    if bad:
        return 1
    if not args.evidence:
        print("note: no --evidence given — admission requires cited, checked evidence "
              "(see skill_library/CLAUDE.md)")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
