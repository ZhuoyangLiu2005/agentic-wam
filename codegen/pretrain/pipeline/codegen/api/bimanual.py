"""Declaration-only YAM bimanual API: the official actor-facing vocabulary.

Semantic-skill verbs (pick/place/grasp/.../handover) over ASPIRE-style
primitives (reference namespace:
``aspire/real/cap/env/real_bimanual_yam/skills.py::make_namespace``):
``get_robot_state``, grasp sampling with candidates + ``select_best_grasp``,
``detect_objects_oneshot``, ``move_arm``/``freespace_move``,
``servo_ee_delta`` (bounded relative motion), side-parameterized
``open_gripper``/``close_gripper``, and ``go_home``. Arm selection is
uniformly the ``side`` parameter (``"left"``/``"right"``; handover uses
``from_side``/``to_side``). Orientation is uniformly rpy DEGREES —
interpretable by the VLM; no quaternion appears anywhere in the surface.
Repeated (CLOSE, OPEN) x N cycles are written as literal ``for _ in range(N)``
loops over close_gripper/rotate/open_gripper — the grounder unrolls literal
loops — and richer wrappers are skill-library territory. Every call returns
the uniform ``ToolResult(success, data, error)`` envelope from ``api/base.py``.

This API is DECLARATION-ONLY: nothing here executes. Gripper effects are
declared in ``bimanual_contracts.py`` (contract form);
``gripper_effects``/``resolve_effects`` are re-exported here for the grounder.
Coverage is asserted at import time so the surface and the contract table
cannot drift apart.

NOTE: no ``from __future__ import annotations`` in this file on purpose —
annotations stay LIVE objects so ``str(inspect.signature(...))`` renders types
unquoted (``ToolResult``, not ``'ToolResult'``) in the API prompt block;
``combined_doc`` shortens this package's qualified names (``ToolResult``,
``GraspCandidate``).
"""
import inspect
from typing import Any

from .base import GraspCandidate, ToolResult, combined_doc
from .bimanual_contracts import (  # noqa: F401 — re-exported for the grounder
    _ARM_PARAM,
    check_api_coverage,
    gripper_effects,
    resolve_effects,
)

_NOT_BOUND = "declaration only — no backend is bound in the inverse code-gen pipeline"


class YamBimanualApi:
    """Bimanual YAM API (declaration-only): perception, semantic skills, and
    side-parameterized primitives. Sides are the strings "left" / "right"."""

    SIDES: tuple[str, ...] = ("left", "right")

    # ------------------------------------------------------------ observation
    def get_robot_state(self) -> ToolResult:
        """Current joint positions, gripper states, and end-effector poses for
        both arms.

        Returns: ToolResult with data {side: {"joint_positions", "gripper_pos",
        "ee_position", "ee_rpy"}} for side in ("left", "right").
        """
        raise NotImplementedError(_NOT_BOUND)

    # ------------------------------------------------------------- perception
    def sample_grasp_pose(self, object_name: str, camera: str = "top", max_grasps: int = 10, top_down_only: bool = False) -> ToolResult:
        """Sample grasp candidates for a named object.

        Returns: ToolResult with data {"grasps": [GraspCandidate, ...]} sorted
        by score; ``width`` predicts where the gripper stalls when closing on
        the object. top_down_only restricts to near-vertical approaches.
        """
        raise NotImplementedError(_NOT_BOUND)

    def select_best_grasp(self, grasp_candidates: list[GraspCandidate], side: str = "right") -> ToolResult:
        """Rank grasp candidates by motion-planning feasibility for ``side`` and
        pick the best reachable one.

        Returns: ToolResult with data {"grasp": GraspCandidate, "index": int}.
        """
        raise NotImplementedError(_NOT_BOUND)

    def detect_objects_oneshot(self, query: str, camera: str = "top") -> ToolResult:
        """Detect named object(s) in one camera frame, returning ALL matches —
        pick the instance you mean, and gate on score before trusting a
        detection.

        Returns: ToolResult with data {"detections": [{"name", "position_3d"
        ((3,) XYZ meters), "score", "box" (image-space xyxy), "extent_3d"
        ((3,) world-frame side lengths, meters)}, ...]} sorted by score.
        Orientation is not estimated; for placement height use extent_3d.
        """
        raise NotImplementedError(_NOT_BOUND)

    # -------------------------------------------------------- semantic skills
    def pick(self, object_name: str, side: str) -> ToolResult:
        """Pick object_name up with the ``side`` arm: grasp, then lift clear of
        the surface; the gripper keeps holding it.

        Use ``grasp`` when the object is not lifted away.
        """
        raise NotImplementedError(_NOT_BOUND)

    def place(self, object_name: str, target_name: str, side: str) -> ToolResult:
        """Place the object held by the ``side`` arm onto/at target_name, then
        release: move over it gripper-down, lower via standoff, open, retreat.
        """
        raise NotImplementedError(_NOT_BOUND)

    def grasp(self, object_name: str, side: str) -> ToolResult:
        """Grasp object_name with the ``side`` arm where it sits (no lift):
        approach from a standoff, close on the sampled grasp pose.

        Also the brace: holding an object steady as a fixture while the other
        arm works. Use ``pick`` when the object is carried away.
        """
        raise NotImplementedError(_NOT_BOUND)

    def release(self, side: str) -> ToolResult:
        """Open the ``side`` gripper and retreat vertically, freeing whatever
        it held.
        """
        raise NotImplementedError(_NOT_BOUND)

    def regrasp(self, object_name: str, side: str) -> ToolResult:
        """Adjust the ``side`` arm's grip on object_name: open, lift slightly,
        re-grasp, close; the object stays where it is.
        """
        raise NotImplementedError(_NOT_BOUND)

    def insert(self, object_name: str, target_name: str, side: str) -> ToolResult:
        """Insert object_name into/onto receptacle target_name with the
        ``side`` arm: grasp it, lift clear, align above the opening, descend,
        release, retreat.
        """
        raise NotImplementedError(_NOT_BOUND)

    def attach(self, object_name: str, target_name: str, side: str) -> ToolResult:
        """Affix the object held by the ``side`` arm onto target_name, then
        release — a place with a seating phase: lower via standoff, press down
        to seat it, open, retreat. The object ends FIXED to the target rather
        than resting on it.
        """
        raise NotImplementedError(_NOT_BOUND)

    def handover(self, object_name: str, from_side: str, to_side: str) -> ToolResult:
        """Pass object_name from the from_side arm to the to_side arm,
        gripper-to-gripper: from_side brings it to a transfer pose, to_side
        grasps it, from_side releases.
        """
        raise NotImplementedError(_NOT_BOUND)

    def rotate(self, object_name: str, angle_deg: float, side: str) -> ToolResult:
        """Rotate the object held by the ``side`` arm in-hand by angle_deg
        about the gripper's approach axis, keeping the end-effector position
        fixed — no regrasp.

        For a repeated screw/twist, loop with a literal count:
        ``for _ in range(n): close_gripper; rotate; open_gripper``.
        """
        raise NotImplementedError(_NOT_BOUND)

    def press(self, target_name: str, side: str) -> ToolResult:
        """Press down on target_name with the ``side`` arm: approach from
        above, command contact past its top surface (no force sensing),
        retreat.
        """
        raise NotImplementedError(_NOT_BOUND)

    # ----------------------------------------------------------------- motion
    def move_arm(self, side: str, target_pos: list[float], target_rpy: list[float]) -> ToolResult:
        """Collision-free move of ONE arm's end-effector to an absolute target
        pose — the preferred single-arm motion call. target_pos is XYZ meters;
        target_rpy is roll-pitch-yaw degrees.
        """
        raise NotImplementedError(_NOT_BOUND)

    def freespace_move(self, left_target_pos: list[float] | None = None, left_target_rpy: list[float] | None = None, right_target_pos: list[float] | None = None, right_target_rpy: list[float] | None = None) -> ToolResult:
        """Collision-free move of BOTH arms in one synchronized call (per-side
        position XYZ meters + rpy degrees). Prefer ``move_arm`` when only one
        arm moves.
        """
        raise NotImplementedError(_NOT_BOUND)

    def servo_ee_delta(self, side: str, delta_pos: list[float], duration_s: float = 1.0) -> ToolResult:
        """Bounded small end-effector translation from the current pose,
        keeping orientation — express what the motion IS ("3 cm up") instead
        of inventing an absolute target. delta_pos is world-frame XYZ meters,
        must be small.
        """
        raise NotImplementedError(_NOT_BOUND)

    def go_home(self) -> ToolResult:
        """Move both arms to the home configuration.
        """
        raise NotImplementedError(_NOT_BOUND)

    # ---------------------------------------------------------------- gripper
    def open_gripper(self, side: str) -> ToolResult:
        """Open the ``side`` gripper (bare primitive — prefer the semantic
        verbs when an object is being let go).
        """
        raise NotImplementedError(_NOT_BOUND)

    def close_gripper(self, side: str) -> ToolResult:
        """Close the ``side`` gripper (bare primitive — prefer the semantic
        verbs when an object is being taken).
        """
        raise NotImplementedError(_NOT_BOUND)

    def functions(self) -> dict[str, Any]:
        return {
            # perception
            "get_robot_state": self.get_robot_state,
            "sample_grasp_pose": self.sample_grasp_pose,
            "select_best_grasp": self.select_best_grasp,
            "detect_objects_oneshot": self.detect_objects_oneshot,
            # semantic
            "pick": self.pick,
            "place": self.place,
            "grasp": self.grasp,
            "release": self.release,
            "regrasp": self.regrasp,
            "insert": self.insert,
            "attach": self.attach,
            "handover": self.handover,
            "rotate": self.rotate,
            "press": self.press,
            # low-level primitives
            "move_arm": self.move_arm,
            "freespace_move": self.freespace_move,
            "servo_ee_delta": self.servo_ee_delta,
            "go_home": self.go_home,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }

    def combined_doc(self) -> str:
        return combined_doc(self.functions())


# API reference handed to the coding agent — reflected from a bare ``__new__``
# instance exactly like the executable original. (ONE abstraction level — the
# old s2 tier keying is gone.)
API_DOC = YamBimanualApi.__new__(YamBimanualApi).combined_doc()

#: Literal ARGUMENT VOCABULARIES, enforced by the static validator (see the
#: ego surface for the rationale): the YAM surface constrains the side params.
ARG_VOCAB = {
    "side": ("left", "right"),
    "from_side": ("left", "right"),
    "to_side": ("left", "right"),
}

# Signatures of the API functions, for resolving a call's side argument BY NAME
# (robust to positional or keyword args).
_API_SIG = {
    name: inspect.signature(fn)
    for name, fn in YamBimanualApi.__new__(YamBimanualApi).functions().items()
}

# Import-time drift guard: every public function must be declared in
# bimanual_contracts.py, either with a gripper contract or as gripper-transparent.
_uncovered = check_api_coverage(YamBimanualApi.__new__(YamBimanualApi).functions())
if _uncovered:
    raise AssertionError(
        f"API functions missing from bimanual_contracts.py (add a Contract or mark "
        f"gripper-transparent): {_uncovered}"
    )
