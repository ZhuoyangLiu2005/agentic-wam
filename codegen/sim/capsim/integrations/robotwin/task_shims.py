"""Supply the state some ``check_success()`` methods read but only ``play_once`` sets (``arm_tag``, ``origin_z``).

Without it the checker raises ``AttributeError`` and every program scores 0. ``origin_z`` is the manipuland's
height at reset; ``arm_tag`` is the arm that last closed on the manipuland, else the nearest TCP, unless
``play_once`` assigned it. Installed by ``RoboTwinHandle.setup()``; add a task in ``checker_shims.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

_SHIMS: dict[str, dict[str, str]] = yaml.safe_load(Path(__file__).with_name("checker_shims.yaml").read_text()) or {}
_MANIPULAND: dict[str, str] = {t: s["manipuland"] for t, s in _SHIMS.items() if s.get("manipuland")}
_ORIGIN_Z: dict[str, str] = {t: s["origin_z"] for t, s in _SHIMS.items() if s.get("origin_z")}
_GRASP_RADIUS = 0.15   # m; TCP-to-manipuland (pose or contact point) distance that counts as grasping it


def _reference_points(actor: Any) -> list[np.ndarray]:
    pts: list[np.ndarray] = []
    try:
        for _, p in actor.iter_contact_points():
            pts.append(np.asarray(p[:3], dtype=np.float64))
    except Exception:  # actors without contact annotations
        pass
    pts.append(np.asarray(actor.get_pose().p, dtype=np.float64))
    return pts


def _tcp(task: Any, hand: str) -> np.ndarray:
    pose = task.robot.get_left_tcp_pose() if hand == "left" else task.robot.get_right_tcp_pose()
    return np.asarray(pose[:3], dtype=np.float64)


def _distance(task: Any, hand: str, actor: Any) -> float:
    tcp = _tcp(task, hand)
    return min(float(np.linalg.norm(tcp - p)) for p in _reference_points(actor))


def _manipuland(task: Any) -> Any:
    return getattr(task, _MANIPULAND[type(task).__name__])


def _arm_tag_get(self: Any) -> Any:
    from envs.utils.action import ArmTag  # type: ignore[import-not-found]

    hand = self.__dict__.get("_capx_arm_tag")          # set explicitly (play_once)
    if hand is None:
        hand = self.__dict__.get("_capx_grasp_hand")   # last close_gripper on the manipuland
    if hand is None:                                   # nearest tool centre point right now
        actor = _manipuland(self)
        hand = min(("left", "right"), key=lambda h: _distance(self, h, actor))
    return ArmTag(hand)


def _arm_tag_set(self: Any, value: Any) -> None:
    self.__dict__["_capx_arm_tag"] = value


def _arm_tag_del(self: Any) -> None:
    self.__dict__.pop("_capx_arm_tag", None)


def install(task: Any) -> None:
    """Supply the play_once-only state for the freshly reset ``task`` (idempotent)."""
    name = type(task).__name__
    if name in _ORIGIN_Z:
        task.origin_z = float(getattr(task, _ORIGIN_Z[name]).get_pose().p[2])
    if name in _MANIPULAND:
        cls = type(task)
        if not isinstance(cls.__dict__.get("arm_tag"), property):
            cls.arm_tag = property(_arm_tag_get, _arm_tag_set, _arm_tag_del)
        for key in ("arm_tag", "_capx_arm_tag", "_capx_grasp_hand"):
            task.__dict__.pop(key, None)


def note_close_gripper(task: Any, hand: str) -> None:
    """Record ``hand`` as the grasping arm when it closes on the manipuland."""
    if type(task).__name__ not in _MANIPULAND:
        return
    try:
        if _distance(task, hand, _manipuland(task)) <= _GRASP_RADIUS:
            task.__dict__["_capx_grasp_hand"] = hand
    except Exception:  # never let bookkeeping break a motion
        pass


def shimmed_tasks() -> tuple[str, ...]:
    return tuple(sorted(set(_MANIPULAND) | set(_ORIGIN_Z)))
