"""Heuristic generator of executable Python-like policy code from an instruction.

Produces a `policy_chunk(obs, state, robot, skills)` function (code-as-policy
style) that calls the mocked skill/robot API. This is *weak* supervision for
bootstrapping the coding modality when no human code annotation exists; it is
deliberately simple, parseable (`ast.parse`), and runnable against the mock API.
Replace with real annotations via `data.use_policy_code` + a real label field.
"""
from __future__ import annotations

import re
from typing import Optional

_VERB_SYNONYMS = {
    "grasp": "grasp", "pick": "grasp", "pick up": "grasp", "take": "grasp", "grab": "grasp", "lift": "grasp",
    "place": "place", "put": "place", "drop": "place", "insert": "place_inside", "release": "place",
    "push": "push", "move": "move", "slide": "push",
    "open": "open", "close": "close",
    "press": "press", "click": "press", "push down": "press",
    "fold": "fold",
    "pour": "pour",
    "handover": "handover", "hand over": "handover",
    "stack": "stack",
}

# crude stopword set so "the cup into the drawer" -> objects [cup, drawer]
_STOP = {
    "the", "a", "an", "into", "in", "on", "onto", "to", "at", "and", "with", "of",
    "robot", "arm", "please", "then", "next", "your", "it", "down", "up", "over",
}


def _detect_verb(text: str) -> str:
    t = text.lower()
    for phrase in ("pick up", "hand over", "push down"):
        if phrase in t:
            return _VERB_SYNONYMS[phrase]
    for w in re.findall(r"[a-z]+", t):
        if w in _VERB_SYNONYMS:
            return _VERB_SYNONYMS[w]
    return "manipulate"


def _detect_objects(text: str) -> list[str]:
    words = re.findall(r"[a-z][a-z\-]*", text.lower())
    objs: list[str] = []
    for w in words:
        if w in _VERB_SYNONYMS or w in _STOP:
            continue
        objs.append(w)
    # dedupe, keep order
    seen, out = set(), []
    for o in objs:
        if o not in seen:
            seen.add(o)
            out.append(o)
    return out[:3] if out else ["object"]


def _q(name: str) -> str:
    return '"' + name.replace('"', "") + '"'


def generate_template_policy_code(instruction: str, task: Optional[str] = None) -> str:
    """Return an ast-parseable `policy_chunk(...)` for the instruction.

    The chunk corresponds to one short-horizon subtask: detect -> (grasp/verify/
    recover) -> act -> verify. Mirrors the skill/robot API in `skill_api.py`.
    """
    text = (instruction or task or "").strip() or "manipulate the object"
    verb = _detect_verb(text)
    objs = _detect_objects(text)
    obj = objs[0]
    target = objs[1] if len(objs) > 1 else None

    L = []
    L.append("def policy_chunk(obs, state, robot, skills):")
    L.append(f"    {obj} = skills.detect({_q(obj)}, obs)")
    if target is not None:
        L.append(f"    {target} = skills.detect({_q(target)}, obs)")

    if verb in ("grasp", "place", "place_inside", "move", "push", "handover", "stack", "pour"):
        # ensure grasp first
        L.append(f"    if not skills.check_attached({obj}, robot.gripper):")
        L.append(f"        grasp_pose = skills.sample_grasp({obj})")
        L.append("        robot.move_to(grasp_pose.pregrasp)")
        L.append("        robot.move_to(grasp_pose.grasp)")
        L.append("        robot.close_gripper()")
        L.append(f"        if not skills.check_attached({obj}, robot.gripper):")
        L.append(f"            return skills.recover_grasp({obj})")

    if verb in ("place", "place_inside", "stack") and target is not None:
        L.append(f"    place_pose = skills.place_inside({obj}, {target})")
        L.append("    robot.move_to(place_pose)")
        L.append("    robot.open_gripper()")
        L.append(f"    return skills.verify_inside({obj}, {target})")
    elif verb == "move" and target is not None:
        L.append(f"    place_pose = skills.sample_place_pose({obj}, {target})")
        L.append("    robot.move_to(place_pose)")
        L.append("    robot.open_gripper()")
        L.append(f"    return skills.visual_diff(obs, skills.track({_q(obj)}, obs))")
    elif verb == "push":
        L.append(f"    push_pose = skills.sample_place_pose({obj}, {target if target else obj})")
        L.append("    robot.move_to(push_pose)")
        L.append("    return skills.visual_diff(obs, obs)")
    elif verb in ("open", "close"):
        L.append(f"    handle_pose = skills.sample_grasp({obj})")
        L.append("    robot.move_to(handle_pose.grasp)")
        L.append("    robot.close_gripper()")
        sign = "-0.2" if verb == "open" else "0.2"
        L.append(f"    robot.delta_pose({sign}, 0.0, 0.0, 0.0, 0.0, 0.0)")
        L.append("    robot.open_gripper()")
        L.append(f"    return skills.visual_diff(obs, obs)")
    elif verb == "press":
        L.append(f"    press_pose = skills.sample_place_pose({obj}, {obj})")
        L.append("    robot.move_to(press_pose)")
        L.append("    robot.delta_pose(0.0, 0.0, -0.05, 0.0, 0.0, 0.0)")
        L.append(f"    return skills.visual_diff(obs, obs)")
    else:
        L.append("    robot.open_gripper()")
        L.append(f"    return skills.verify_inside({obj}, {target if target else obj})")

    return "\n".join(L) + "\n"
