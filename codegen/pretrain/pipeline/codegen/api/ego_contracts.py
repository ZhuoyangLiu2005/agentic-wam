"""Contact-event contracts for the egocentric hand API (EgoHandApi).

For each API function, the ORDERED net hand open/close transitions it
produces (``Contract``/``GripEffect`` form) AS THE APERTURE DETECTOR OBSERVES THEM
(``codegen.grounding.aperture`` — the Qwen-RoboManip retargeted-width proxy).
Same event alphabet as the YAM surface ("close"/"open" on "left"/"right"), so
the grounder (``codegen.program.effects``) is shared unchanged.

``arm_ref`` semantics:
  * ``"side"``           -> the call's ``side`` parameter (all single-hand calls)
  * ``"from"`` / ``"to"`` -> handover's ``from_side``/``to_side`` parameters
    (all resolve via ``_ARM_PARAM``)

THREE contracts deliberately differ from the YAM surface — a human hand is the
sensor here, and the YAM contract describes a gripper:
  * ``handover``: a HUMAN hand-to-hand pass reads as (to CLOSE, from OPEN) —
    the receiver secures the object before the giver lets go; there is no
    initial giver-grasp event because the giver already holds the object.
  * ``insert``: OPEN only — an ego "Insert X into Y" subtask begins with X
    already in the hand (the pick was the previous subtask), so the verb
    asserts the seating's letting-go, not YAM's full CLOSE,OPEN
    grasp-carry-release composition.
  * ``press``: CLOSE -> OPEN — a human press IS an aperture cycle (the hand
    contacts the target and comes off it), where the YAM gripper presses with
    a closed jaw and is gripper-transparent.

Any non-contract arguments are contract-transparent detail. Every table entry
is STATIC; every function without a contract is declared transparent
explicitly; ``ego.py`` asserts full coverage at import time.
"""
from __future__ import annotations

from .contracts import Contract, GripEffect

#: effect arm_ref -> the call parameter that names the hand.
_ARM_PARAM = {"side": "side", "from": "from_side", "to": "to_side"}

CONTRACTS: dict[str, Contract] = {
    # explicit primitives
    "close_hand": Contract(grip=(GripEffect("side", "close"),)),
    "open_hand": Contract(grip=(GripEffect("side", "open"),)),
    # semantic contact verbs
    "pick": Contract(grip=(GripEffect("side", "close"),)),
    "grasp": Contract(grip=(GripEffect("side", "close"),)),
    "place": Contract(grip=(GripEffect("side", "open"),)),
    "release": Contract(grip=(GripEffect("side", "open"),)),
    "regrasp": Contract(grip=(GripEffect("side", "open"), GripEffect("side", "close"))),
    "insert": Contract(grip=(GripEffect("side", "open"),)),   # seat the HELD object, let go
    "attach": Contract(grip=(GripEffect("side", "open"),)),   # place + seating -> the release open
    # receiver secures the object before the giver lets go (giver already holds it)
    "handover": Contract(grip=(GripEffect("to", "close"), GripEffect("from", "open"))),
    "press": Contract(grip=(GripEffect("side", "close"), GripEffect("side", "open"))),
}

# Explicitly contact-transparent (produce no aperture transition).
GRIPPER_TRANSPARENT = frozenset({
    "move_arm", "goto_relative", "rotate",
})


def contract(name: str, args: dict | None = None) -> Contract | None:
    """Contract of ONE API call, by function name (None = contact-transparent).

    ``args`` is accepted for interface stability with argument-dependent
    contract tables; every ego contract is static."""
    return CONTRACTS.get(name)


def gripper_effects(name: str, args: dict | None = None) -> list[tuple[str, str]] | None:
    """Grounder-facing view: ordered ``[(arm_ref, kind)]`` of ONE call, or None
    for contact-transparent functions."""
    c = contract(name, args)
    if c is None:
        return None
    return [(e.arm_ref, e.kind) for e in c.grip]


def resolve_effects(name: str, bound_args: dict | None = None) -> list[tuple[str, str]] | None:
    """Fully resolved effects of ONE call: ordered ``[(side_literal, kind)]``,
    or None for contact-transparent functions — the same resolver interface as
    the YAM surface (``bimanual_contracts.resolve_effects``).

    Parameter refs resolve through ``_ARM_PARAM`` against the call's bound
    arguments; an unresolvable ref comes back as the ref name itself so the
    caller treats it as a diagnostic rather than crash."""
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
