"""The shared contract DATACLASSES — the declaration form every embodiment's
contract table (``bimanual_contracts.py``, ``ego_contracts.py``) is written in.

A ``Contract`` declares the ordered, sensor-observable effects of one API
call; ``GripEffect.arm_ref`` names the call parameter that selects the arm
(resolved by each table's ``resolve_effects``). Future channels (motion
signatures, contact-close discrimination, held-state pre/post) slot into
``Contract`` without changing the resolver interface.

(The ASPIRE-verbatim draft tables that used to live here are retired with the
draft surface — see ``_attic/contracts_aspire_draft.py``.)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GripEffect:
    """One sensor-observable gripper/contact transition produced by an API call.

    arm_ref: parameter name in the call whose value ("left"|"right") is the arm
             (or a fixed "left"/"right" literal).
    kind:    "open" | "close".
    """

    arm_ref: str
    kind: str


@dataclass(frozen=True)
class Contract:
    """Declared, ordered effect of one API function. Grip channel only (for now)."""

    grip: tuple[GripEffect, ...] = field(default_factory=tuple)
