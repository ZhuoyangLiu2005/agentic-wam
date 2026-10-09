"""Gripper-event contracts for the YAM bimanual API (YamBimanualApi).

For each API function, the ORDERED net gripper transitions it produces AS
THE SENSOR OBSERVES THEM (``Contract``/``GripEffect`` form). This lets the
grounding pass align observed open/close events to the code even when the
events are HIDDEN inside semantic calls (``pick``/``place``/...), not just
explicit ``open_gripper``/``close_gripper``.

``arm_ref`` semantics:
  * ``"side"``           -> the call's ``side`` parameter (all single-arm calls)
  * ``"from"`` / ``"to"`` -> handover's ``from_side``/``to_side`` parameters
    (all resolve via ``_ARM_PARAM``; ``"left"``/``"right"`` would name a FIXED
    arm — no current function uses one)

The preparatory ``_open`` inside grasp/pick is a no-op when the gripper is
already open (the usual case), so it is NOT listed — the table records only the
transitions ``grip_transitions`` actually detects. Every table entry is
STATIC; repeated (CLOSE, OPEN) cycles are literal program loops over
close_gripper/rotate/open_gripper, which the grounder unrolls. Every function
without a contract is declared gripper-transparent explicitly; ``bimanual.py``
asserts full coverage at import time.
"""
from __future__ import annotations

from .contracts import Contract, GripEffect

#: effect arm_ref -> the call parameter that names the arm ("left"/"right" are fixed).
_ARM_PARAM = {"side": "side", "from": "from_side", "to": "to_side"}

CONTRACTS: dict[str, Contract] = {
    # explicit primitives (side-parameterized, ASPIRE-style)
    "close_gripper": Contract(grip=(GripEffect("side", "close"),)),
    "open_gripper": Contract(grip=(GripEffect("side", "open"),)),
    # semantic skills (gripper events bundled inside)
    "grasp": Contract(grip=(GripEffect("side", "close"),)),
    "pick": Contract(grip=(GripEffect("side", "close"),)),
    "place": Contract(grip=(GripEffect("side", "open"),)),
    "release": Contract(grip=(GripEffect("side", "open"),)),
    "attach": Contract(grip=(GripEffect("side", "open"),)),   # place + press -> the release open
    "regrasp": Contract(grip=(GripEffect("side", "open"), GripEffect("side", "close"))),
    "insert": Contract(grip=(GripEffect("side", "close"), GripEffect("side", "open"))),
    "handover": Contract(grip=(GripEffect("from", "close"),
                               GripEffect("to", "close"),
                               GripEffect("from", "open"))),
}

# Explicitly gripper-transparent (produce no gripper transition).
GRIPPER_TRANSPARENT = frozenset({
    "get_robot_state",
    "sample_grasp_pose", "select_best_grasp", "detect_objects_oneshot",
    "rotate", "press",
    "move_arm", "freespace_move", "servo_ee_delta", "go_home",
})


def contract(name: str, args: dict | None = None) -> Contract | None:
    """Contract of ONE API call, by function name (None = gripper-transparent).

    ``args`` is accepted for interface stability with argument-dependent
    contract tables; every contract here is static."""
    return CONTRACTS.get(name)


def gripper_effects(name: str, args: dict | None = None) -> list[tuple[str, str]] | None:
    """Grounder-facing view: ordered ``[(arm_ref, kind)]`` of ONE call, or None
    for gripper-transparent functions."""
    c = contract(name, args)
    if c is None:
        return None
    return [(e.arm_ref, e.kind) for e in c.grip]


def resolve_effects(name: str, bound_args: dict | None = None) -> list[tuple[str, str]] | None:
    """Fully resolved effects of ONE call: ordered ``[(arm_literal, kind)]``, or
    None for gripper-transparent functions — same resolver interface as
    ``ego_contracts.resolve_effects``.

    Fixed refs ("left"/"right") pass through; parameter refs resolve through
    ``_ARM_PARAM`` against the call's bound arguments. An unresolvable ref (the
    side argument is missing or non-literal) comes back as the ref name itself
    so the caller can treat it as a diagnostic rather than crash."""
    c = contract(name, bound_args)
    if c is None:
        return None
    out = []
    for e in c.grip:
        if e.arm_ref in ("left", "right"):
            out.append((e.arm_ref, e.kind))
        else:
            v = (bound_args or {}).get(_ARM_PARAM[e.arm_ref])
            out.append((v if isinstance(v, str) else e.arm_ref, e.kind))
    return out


def check_api_coverage(public_functions) -> list[str]:
    """Names of API functions with NO declaration (no contract, not marked
    transparent). Empty = full coverage."""
    declared = set(CONTRACTS) | set(GRIPPER_TRANSPARENT)
    return sorted(n for n in public_functions if n not in declared)
