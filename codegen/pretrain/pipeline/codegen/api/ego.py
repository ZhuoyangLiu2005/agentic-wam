"""Declaration-only egocentric hand API: the actor-facing vocabulary for
egocentric human-hand demonstration data.

This is the ego embodiment's OWN surface — disjoint from the YAM surface in
``bimanual.py`` but aligned with it wherever the concepts coincide: the same
semantic verb set (pick/place/grasp/release/regrasp/insert/attach/handover/
rotate/press) and hand selection uniformly via the ``side`` parameter
(``"left"``/``"right"``; handover uses ``from_side``/``to_side``). ``hold`` is
NOT a verb (brace = ``grasp`` + intent comment), and repeated twist cycles are
literal program loops over ``close_hand``/``rotate``/``open_hand``.

Deliberate divergences from the YAM surface (a human hand is not a gripper):
  * NO perception layer (no detections, no grasp sampling, no robot state): a
    human demonstration contains no perception calls to reconstruct; objects
    are named in plain language and spatial intent is expressed through the
    semantic motion primitives.
  * NO coordinates anywhere: ``move_arm`` shares the YAM name but takes a
    semantic direction, not an absolute pose; ``goto_relative`` positions by
    relation to a named object.
  * ``rotate`` keeps a semantic ``axis`` parameter (pour vs flip vs screw is
    frame-observable geometry; the YAM rotate axis is the gripper approach
    axis, which a hand does not have) and covers BOTH a held object and one
    pushed/spun without a hold.
  * ``press``/``insert`` contracts differ from YAM — see ``ego_contracts.py``.
  * NO return values: every call returns ``None``. The YAM ``ToolResult``
    envelope exists for real perception/state data flow; ego has none, the
    corpus never bound a return (0 of 2734 lines, e1–e15), and a fictional
    ``success`` invites control flow the static grounder cannot resolve.

DOCSTRING DISCIPLINE (these docstrings ARE the actor prompt, via
``combined_doc``): one sentence of precise semantics, plus a boundary against
an adjacent verb only where genuine ambiguity exists. NO use-case
enumerations — funnelling maneuvers into a verb by example suppresses the
divergent-spelling signal that skill mining and API evolution feed on. No
restating of signatures or of vocabularies carried by CONV/ARG_VOCAB.

DECLARATION-ONLY: nothing executes; every method raises. Contact effects are
declared in ``ego_contracts.py``; coverage is asserted at import time.
"""
import inspect
from typing import Any

from .base import combined_doc
from .ego_contracts import (  # noqa: F401 — re-exported for the grounder
    _ARM_PARAM,
    check_api_coverage,
    gripper_effects,
    resolve_effects,
)

_NOT_BOUND = "declaration only — no backend is bound in the inverse code-gen pipeline"

#: Semantic argument vocabularies (egocentric body/camera frame; resolved by a
#: future backend from perception, never from absolute coordinates).
MOVE_DIRECTIONS = ("toward_body", "away_from_body", "up", "down", "left", "right")
ROTATE_AXES = ("object_long", "object_symmetry", "vertical", "lateral")
RELATIONS = ("above", "below", "left_of", "right_of", "in_front_of", "behind",
             "inside", "against")

#: Literal ARGUMENT VOCABULARIES, enforced by the static validator: a literal
#: value bound to one of these params must be in-vocabulary (an out-of-
#: vocabulary literal would silently garble grounding or motion semantics).
#: These are NOT contracts — contracts declare sensor-observable EFFECTS;
#: these constrain what the actor may SAY. Single source for the machine
#: check; the actor sees the values in CONV.
ARG_VOCAB = {
    "side": ("left", "right"),
    "from_side": ("left", "right"),
    "to_side": ("left", "right"),
    "direction": MOVE_DIRECTIONS,
    "relation": RELATIONS,
    "axis": ROTATE_AXES,
}


class EgoHandApi:
    """Bimanual human-hand API (declaration-only): semantic verbs + hand
    primitives + semantic motion. Sides are the strings "left" / "right"."""

    SIDES: tuple[str, ...] = ("left", "right")

    # -------------------------------------------------------- semantic verbs
    def pick(self, object_name: str, side: str) -> None:
        """Take object_name in the ``side`` hand and lift it clear of its
        support; the hand keeps holding it.

        Use ``grasp`` when the hand closes on something it will NOT carry.
        """
        raise NotImplementedError(_NOT_BOUND)

    def grasp(self, object_name: str, side: str) -> None:
        """Close the ``side`` hand on object_name where it sits, without
        lifting it away.

        Also the brace: holding an object steady as a fixture while the other
        hand works. Use ``pick`` when the object is carried away.
        """
        raise NotImplementedError(_NOT_BOUND)

    def place(self, object_name: str, target_name: str, side: str) -> None:
        """Set the object held by the ``side`` hand down onto/at target_name,
        then let go ('table' when the video shows no finer target).
        """
        raise NotImplementedError(_NOT_BOUND)

    def release(self, side: str) -> None:
        """Open the ``side`` hand, freeing whatever it held, with no placement
        intent — letting go in place, or dropping.
        """
        raise NotImplementedError(_NOT_BOUND)

    def regrasp(self, object_name: str, side: str) -> None:
        """Momentarily open and re-close the ``side`` hand on object_name — a
        grip adjustment; the object stays where it is.
        """
        raise NotImplementedError(_NOT_BOUND)

    def insert(self, object_name: str, target_name: str, side: str) -> None:
        """Seat the object ALREADY HELD by the ``side`` hand into receptacle
        target_name, then let go.

        Insert asserts the letting-go: when the hand keeps holding after
        seating, write the seating as ``goto_relative(..., 'inside')`` instead.
        """
        raise NotImplementedError(_NOT_BOUND)

    def attach(self, object_name: str, target_name: str, side: str) -> None:
        """Affix the object held by the ``side`` hand onto target_name, then
        let go — a place whose seating needs contact force, so the object ends
        FIXED to the target rather than resting on it.
        """
        raise NotImplementedError(_NOT_BOUND)

    def handover(self, object_name: str, from_side: str, to_side: str) -> None:
        """Pass object_name between hands: the to_side hand secures it BEFORE
        the from_side hand lets go.
        """
        raise NotImplementedError(_NOT_BOUND)

    def rotate(self, object_name: str, axis: str, angle_deg: float, side: str) -> None:
        """Turn object_name about a semantic axis with the ``side`` hand —
        held in the grasp OR pushed/spun without a hold; the grip state does
        not change either way.

        For a repeated screw/twist, loop with a literal count:
        ``for _ in range(n): close_hand; rotate; open_hand``.
        """
        raise NotImplementedError(_NOT_BOUND)

    def press(self, target_name: str, side: str) -> None:
        """Press on target_name with the ``side`` hand: contact it and come
        off it — a close-then-open contact pair.
        """
        raise NotImplementedError(_NOT_BOUND)

    # ------------------------------------------------------- hand primitives
    def close_hand(self, side: str) -> None:
        """Close the ``side`` hand (bare primitive — prefer ``pick``/``grasp``
        when an object is being taken).
        """
        raise NotImplementedError(_NOT_BOUND)

    def open_hand(self, side: str) -> None:
        """Open the ``side`` hand (bare primitive — prefer ``place``/
        ``release`` when an object is being let go).
        """
        raise NotImplementedError(_NOT_BOUND)

    # ------------------------------------------- contact-transparent motion
    def move_arm(self, side: str, direction: str, distance: float | None = None) -> None:
        """Translate the ``side`` hand (and whatever it holds/touches) along a
        semantic direction; the grip state does not change.

        For displacement whose meaning is the direction itself, with no
        perceivable target object.
        """
        raise NotImplementedError(_NOT_BOUND)

    def goto_relative(self, side: str, object_name: str, relation: str, clearance: float | None = None) -> None:
        """Position the ``side`` hand (and whatever it holds) at a semantic
        spatial relation to object_name; the grip state does not change.

        For targets defined by another object's geometry (a part may be named:
        'bottle neck') — never fake a relation with numeric offsets.
        """
        raise NotImplementedError(_NOT_BOUND)

    def functions(self) -> dict[str, Any]:
        # NOTE: dict order = the order combined_doc() presents the API to the
        # coding agent: semantic verbs first (the preferred vocabulary), then
        # hand primitives, then the motion layer.
        return {
            # semantic
            "pick": self.pick,
            "grasp": self.grasp,
            "place": self.place,
            "release": self.release,
            "regrasp": self.regrasp,
            "insert": self.insert,
            "attach": self.attach,
            "handover": self.handover,
            "rotate": self.rotate,
            "press": self.press,
            # low-level primitives
            "close_hand": self.close_hand,
            "open_hand": self.open_hand,
            "move_arm": self.move_arm,
            "goto_relative": self.goto_relative,
        }

    def combined_doc(self) -> str:
        return combined_doc(self.functions())


# API reference handed to the coding agent — reflected from a bare ``__new__``
# instance exactly like the YAM surface.
API_DOC = EgoHandApi.__new__(EgoHandApi).combined_doc()

# Signatures of the API functions, for resolving a call's side argument BY NAME
# (robust to positional or keyword args).
_API_SIG = {
    name: inspect.signature(fn)
    for name, fn in EgoHandApi.__new__(EgoHandApi).functions().items()
}

# Import-time drift guard: every public function must be declared in
# ego_contracts.py, either with a contact contract or as contact-transparent.
_uncovered = check_api_coverage(EgoHandApi.__new__(EgoHandApi).functions())
if _uncovered:
    raise AssertionError(
        f"API functions missing from ego_contracts.py (add a Contract or mark "
        f"contact-transparent): {_uncovered}"
    )
