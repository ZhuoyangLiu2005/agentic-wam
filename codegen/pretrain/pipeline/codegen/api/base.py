"""Shared API plumbing for both actor-facing surfaces.

Both surfaces (``bimanual.py``, ``ego.py``) build on the pieces here, so the
embodiments speak ONE format:

  * ``ToolResult`` — the uniform result envelope on every API call (same shape
    as the ASPIRE real-robot tools: success flag, data payload, error string).
  * ``combined_doc()`` — renders ``{name: callable}`` into the standardized API
    prompt block (name+signature line, then the docstring indented under
    ``Doc:``).
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Result envelope for every API call (same shape as the real-robot tools)."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class GraspCandidate:
    """One grasp proposal from grasp-pose sampling.

    position: (3,) XYZ meters, world frame.  rpy: (3,) roll-pitch-yaw degrees.
    score: grasp quality in [0, 1].  width: expected gripper width (m) at grasp.
    """

    position: tuple[float, float, float]
    rpy: tuple[float, float, float]
    score: float
    width: float


def combined_doc(functions: dict[str, Callable[..., Any]]) -> str:
    """Render {name: callable} into the standardized API prompt block.

    Live class annotations render module-qualified; classes from this shared
    module are canonicalized to their short names (``ToolResult``,
    ``GraspCandidate``) for the prompt."""
    prefix = f"{ToolResult.__module__}."
    lines: list[str] = []
    for name, fn in functions.items():
        try:
            sig = str(inspect.signature(fn)).replace(prefix, "")
        except Exception:
            sig = "(…)"
        doc = inspect.getdoc(fn) or ""
        lines.append(f"{name}{sig}")
        if doc:
            lines.append("  Doc:")
            lines.extend(f"    {ln}" for ln in doc.splitlines())
        lines.append("")
    return "\n".join(lines).strip()
