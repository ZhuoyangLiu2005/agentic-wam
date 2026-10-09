"""ASPIRE-style composed-function skill library for the YAM inverse-codegen pipeline.

Port of the settled dex design (cap-x `dex_skills.py`, user directive
2026-07-19): a skill is a plain Python function whose BODY is core
`YamTwoArmApi` verbs plus comments — the API itself never grows; richer verbs
are skills the coordinator admits into the shared library. Each admitted skill
lives at

    yam_agent_skills/skills/<name>/skill.py    (the composed function)
    yam_agent_skills/skills/<name>/SKILL.md    (evidence + when-to-use doc)

The function's first parameter is ``robot``; generated programs call skills as
plain functions, e.g. ``stabilize_and_act(robot, 'box', 'left')``. Programs only
IMPORT skills (one short header line), so:

  * programs stay self-contained and always compile stand-alone;
  * the grounding step resolves imported skill calls from the library registry
    and derives a call's gripper effects by EXPANDING its body at the call site
    (nothing is hand-declared; see vlm_codegen.effective_api_calls);
  * the coder prompt shows the same source the validator accepts.

Validation is strict on purpose: one top-level function per file, name equal to
the skill dir (kebab -> underscores), first param ``robot``, no imports, and
every call inside must be ``robot.<core-fn>`` with arguments its signature
accepts and legal literal arm values where given. Simple assignments of query
results (``pos, quat = robot.get_object_pose(...)``) are fine — perception
grounding is how a skill's motion phase stays semantic instead of numeric. A
skill may not call another skill (one expansion level keeps grounding exact).

Skills carry a TWO-TIER validation status (see each SKILL.md): the gripper
contract (the body's close/open sequence) is evidence-grounded by inverse
reconciliation (grip_pins); any motion scaffold (perception + goto_pose lines)
is "proposed" until executed — per the 2026-07-19 decision, sim execution is
out of scope for data-gen, so tier 2 stays advisory.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

from agenticwam.coding.skill_api_yam import FUNCTIONS

ARMS = ("left", "right")
_ARM_PARAMS = ("arm", "from_arm", "to_arm")
DEFAULT_LIB = Path(__file__).resolve().parent / "skills" / "skills"
_API_SIG = {name: inspect.signature(fn)
            for name, fn in FUNCTIONS.items()}


def validate_skill_source(src: str, name: str | None = None) -> list[str]:
    """Problems with a candidate skill source; [] = valid."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} at line {e.lineno}"]
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    others = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.Expr))]
    probs = []
    if len(fns) != 1:
        probs.append(f"skill file must define exactly ONE function (found {len(fns)})")
        return probs
    fn = fns[0]
    if others:
        probs.append("only the function definition (and a module docstring) is allowed")
    if name and fn.name != name.replace("-", "_"):
        probs.append(f"function name {fn.name!r} must equal the skill dir name "
                     f"{name!r} (kebab -> underscores)")
    if not fn.args.args or fn.args.args[0].arg != "robot":
        probs.append("first parameter must be `robot`")
    for node in ast.walk(fn):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            probs.append("imports are not allowed in a skill body")
        if isinstance(node, ast.Call):
            f = node.func
            if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                    and f.value.id == "robot"):
                probs.append(f"line {node.lineno}: skills may only call robot.<api> "
                             f"(no nested skills, no free functions)")
                continue
            if f.attr not in _API_SIG:
                probs.append(f"line {node.lineno}: robot.{f.attr} is not a core API function")
                continue
            lit = lambda n: n.value if isinstance(n, ast.Constant) else None  # noqa: E731
            try:
                bound = dict(_API_SIG[f.attr].bind_partial(
                    *[lit(a) for a in node.args],
                    **{kw.arg: lit(kw.value) for kw in node.keywords if kw.arg}).arguments)
            except TypeError:
                probs.append(f"line {node.lineno}: robot.{f.attr}(...) — arguments do not "
                             f"fit its signature")
                continue
            for p in _ARM_PARAMS:
                v = bound.get(p)
                if isinstance(v, str) and v not in ARMS:  # skill params flow through as None
                    probs.append(f"line {node.lineno}: {p}={v!r} not in {ARMS}")
    return probs


def load_skills(lib: Path = DEFAULT_LIB) -> list[dict]:
    """Valid composed-function skills: [{name, dir, source, doc, params}].

    Invalid skill.py files are skipped with a stderr warning (doc-only skill
    dirs — no skill.py — are silently ignored; they are patterns for the
    coordinator, not callable vocabulary)."""
    skills = []
    if not lib.is_dir():
        return skills
    for d in sorted(p for p in lib.iterdir() if p.is_dir()):
        py = d / "skill.py"
        if not py.exists():
            continue
        src = py.read_text()
        probs = validate_skill_source(src, name=d.name)
        if probs:
            print(f"[yam_skills] SKIPPING invalid skill {d.name}: {probs[0]}",
                  file=sys.stderr)
            continue
        fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef))
        skills.append({
            "name": fn.name,          # callable name (underscores)
            "dir": d.name,            # library dir (kebab-case)
            "source": src.strip(),
            "doc": ast.get_docstring(fn) or "",
            "params": [a.arg for a in fn.args.args],
        })
    return skills


def skills_header(skills: list[dict]) -> list[str]:
    """Program-header lines IMPORTING the skills (kept short on purpose — the
    full sources live in yam_agent_skills/skills/*/skill.py and are resolved by
    the grounding step/validator from the library, not from the program text)."""
    if not skills:
        return []
    names = ", ".join(s["name"] for s in skills)
    return [f"from yam_agent_skills import {names}  "
            f"# composed skills (source: yam_agent_skills/skills/<name>/skill.py)"]


def library_defs_ast(lib: Path = DEFAULT_LIB) -> dict:
    """{fn_name: ast.FunctionDef} for every valid library skill — the registry
    the grounding step and validator use to resolve imported skill calls."""
    out = {}
    for s in load_skills(lib):
        out[s["name"]] = next(n for n in ast.parse(s["source"]).body
                              if isinstance(n, ast.FunctionDef))
    return out


def skills_prompt_doc(skills: list[dict]) -> str:
    """Prompt section describing the callable skills ('' if none). Includes the
    full source: programs only IMPORT skills, so the prompt is where the coder
    learns each skill's semantics and body."""
    if not skills:
        return ""
    parts = ["SKILL LIBRARY (imported at the top of the program — call them as "
             f"plain functions, robot first, e.g. {skills[0]['name']}(robot, ...)); "
             "PREFER a skill over spelling out its sequence when it matches what "
             "you see:"]
    for s in skills:
        parts.append("```python\n" + s["source"] + "\n```")
    return "\n".join(parts)


# --- back-compat: v1 skill vocabulary (superseded by the ASPIRE library above; kept for
# the archived pre-ASPIRE generators + the optional skill-conditioning head) ---
from agenticwam.coding._legacy.skill_vocab_v1 import (  # noqa: E402,F401
    SkillSpec, REDUCED_SKILLS, NUM_SKILLS, SKILL_NAMES, NAME_TO_ID, ID_TO_NAME,
    skill_text, skill_catalog, SkillLibrary,
)
