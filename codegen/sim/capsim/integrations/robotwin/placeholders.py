"""Instruction placeholders for a collected episode without running RoboTwin's expert.

The ``self.info["info"] = {...}`` literal of a task's ``play_once`` reads only scene attributes and the arm
variable, so it is evaluated on the live task with the arm bound to the one the program used. Literals with
several arm variables need the expert's ordering (``needs_expert=True``).
"""
from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from typing import Any

_ARM_NAME = re.compile(r"^(arm_tag|arm|hand|arm_info)$")
_ARM_LIKE = re.compile(r"(arm_tag\d*|_arm_tag|arm_info|^arm$|^hand$)")


def info_dict_source(task_obj: Any) -> ast.Dict | None:
    """The `self.info["info"] = {...}` literal of the task class's play_once, or None."""
    try:
        src = inspect.getsource(type(task_obj).play_once)
    except (OSError, TypeError, AttributeError):
        return None
    try:
        tree = ast.parse(inspect.cleandoc(src) if not src.startswith("def") else src)
    except SyntaxError:
        tree = ast.parse("\n".join(l[4:] if l.startswith("    ") else l for l in src.splitlines()))
    node = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(ast.unparse(t) == "self.info['info']" for t in n.targets) \
                and isinstance(n.value, ast.Dict):
            node = n.value
    return node


def arm_variables(d: ast.Dict) -> list[str]:
    names = []
    for v in d.values:
        for n in ast.walk(v):
            if isinstance(n, ast.Name) and _ARM_LIKE.search(n.id) and n.id not in names:
                names.append(n.id)
    return names


def evaluate(task_obj: Any, hands: list[str]) -> dict[str, Any]:
    """Evaluate the info literal with the arm bound to ``hands[0]`` (the program's first grasp)."""
    d = info_dict_source(task_obj)
    if d is None:
        return {"placeholders": None, "needs_expert": False, "arm_vars": [], "error": "no info dict in play_once"}
    arms = arm_variables(d)
    out = {"placeholders": None, "needs_expert": len(arms) > 1, "arm_vars": arms, "error": None}
    if out["needs_expert"]:
        return out
    ns: dict[str, Any] = {"self": task_obj, "str": str}
    if arms:
        if not hands:
            out["error"] = "program issued no grasp: arm placeholder unknown"
            return out
        ns[arms[0]] = hands[0]
    try:
        out["placeholders"] = {str(k): str(v) for k, v in eval(  # noqa: S307 (task source, attribute reads only)
            compile(ast.Expression(d), "<info>", "eval"), {"__builtins__": {}}, ns).items()}
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# Program templates name per-episode objects as "{B}" in string literals; materialize fills in the noun
# in place (line numbers preserved). Arm keys ({a}) are never filled.
_ASSET_RE = re.compile(r"^(\d{3})_([A-Za-z][\w-]*)/base(\d+)$")
_SLOT_RE = re.compile(r"\{([A-Z])\}")
# Closed compounds spelled the way perception grounds them; others just open '-'/'_' into spaces.
def _asset_nouns() -> dict:
    """object_vocab.yaml `asset_nouns` (read directly: this module is also loaded by path, without capsim)."""
    import yaml
    return yaml.safe_load((Path(__file__).with_name("object_vocab.yaml")).read_text())["asset_nouns"]


def canonical_noun(asset: str) -> str:
    """'047_mouse/base1' -> 'mouse'; '112_tea-box/base0' -> 'tea box'; '072_electronicscale/base2' -> 'electronic scale'."""
    m = _ASSET_RE.match(asset.strip())
    if not m:
        raise ValueError(f"not an asset path: {asset!r}")
    stem = m.group(2)
    return _asset_nouns().get(stem, stem.replace("-", " ").replace("_", " ")).lower()


def object_slots(task_obj: Any) -> dict[str, str]:
    """The info literal's asset-path values (e.g. {'{B}': '048_stapler/base1'}), arms bound to a dummy."""
    d = info_dict_source(task_obj)
    if d is None:
        return {}
    ns: dict[str, Any] = {"self": task_obj, "str": str}
    for a in arm_variables(d):
        ns[a] = "left"
    out = {}
    for kn, vn in zip(d.keys, d.values):      # per value: self.arm_tag may not exist before play_once
        try:
            k = eval(compile(ast.Expression(kn), "<info>", "eval"), {"__builtins__": {}}, ns)  # noqa: S307
            v = eval(compile(ast.Expression(vn), "<info>", "eval"), {"__builtins__": {}}, ns)  # noqa: S307
        except Exception:  # noqa: BLE001
            continue
        if _ASSET_RE.match(str(v)):
            out[str(k)] = str(v)
    return out


def template_slots(text: str) -> list[str]:
    """The object placeholder keys a program template uses inside its string literals, e.g. ['{B}']."""
    keys = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for m in _SLOT_RE.finditer(node.value):
                if m.group(0) not in keys:
                    keys.append(m.group(0))
    return keys


def materialize(text: str, slots: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Fill object placeholders -> (program text, {key: noun}); KeyError on a slot the episode lacks."""
    keys = template_slots(text)
    nouns = {}
    for k in keys:
        if k not in slots:
            raise KeyError(f"program uses {k} but the task's info literal has no object for it (slots: {sorted(slots)})")
        nouns[k] = canonical_noun(slots[k])
    out = text
    for k, noun in nouns.items():
        out = out.replace(k, noun)
    return out, nouns


def _hand_of(r: dict) -> str | None:
    h = r.get("hand") or (r.get("top_locals") or {}).get("hand")     # top_locals: records without a `hand` field
    return h if h in ("left", "right") else None


def robotwin_root(robotwin_root: str | None = None) -> Path:
    """The RoboTwin checkout (argument, $CAPX_ROBOTWIN_ROOT, package symlink); raises rather than degrade to 'off'."""
    import os
    here = Path(__file__).resolve()
    cands = [robotwin_root, os.environ.get("CAPX_ROBOTWIN_ROOT"), str(here.parents[2] / "third_party" / "RoboTwin")]
    for c in cands:
        if c and (Path(c) / "envs").is_dir():
            return Path(c)
    raise FileNotFoundError("RoboTwin checkout not found (tried " + ", ".join(str(c) for c in cands if c)
                            + "); source env.sh or set CAPX_ROBOTWIN_ROOT")


def info_dict_from_file(task_name: str, robotwin_root_: str | None = None) -> ast.Dict | None:
    """The info literal from envs/<task>.py without a simulator; None only when the task has none."""
    src = robotwin_root(robotwin_root_) / "envs" / f"{task_name}.py"
    if not src.is_file():
        raise FileNotFoundError(f"no RoboTwin task source {src}")
    tree = ast.parse(src.read_text())
    node = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(ast.unparse(t) == "self.info['info']" for t in n.targets) and isinstance(n.value, ast.Dict):
            node = n.value
    return node


def recommended_mode(task_name: str, robotwin_root_: str | None = None) -> tuple[str, list[str]]:
    """('static' | 'expert' | 'off', arm variables): the record_instructions mode for this task."""
    d = info_dict_from_file(task_name, robotwin_root_)
    if d is None:
        return "off", []
    arms = arm_variables(d)
    return ("expert" if len(arms) > 1 else "static"), arms


def program_hands(records: list[dict]) -> list[str]:
    """Arms of the program's grasps (close_gripper) in order; falls back to the first motion's arm."""
    hands = [h for r in records if r.get("api") == "close_gripper" and (h := _hand_of(r))]
    if not hands:
        hands = [h for r in records if (h := _hand_of(r))][:1]
    return hands


if __name__ == "__main__":                       # python capsim/integrations/robotwin/placeholders.py <task>...
    import sys
    for t in sys.argv[1:]:
        mode, arms = recommended_mode(t)
        d = info_dict_from_file(t)
        print(f"{t:26s} record_instructions={mode:7s} arm_vars={arms} placeholders={[ast.unparse(k) for k in d.keys] if d else None}")
