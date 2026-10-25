"""30 Hz chunk checks for the vega+Sharpa loop -- the counterpart of YAM's `yam_station.robot.safety.SafetyGuard`.

The motion limits are NOT here: the vendored 300 Hz thread (vega/arm_hand_control.SmoothingAndSafetyManager)
already caps arm speed at 0.4 x URDF velocity, clips every joint to its URDF range and refuses colliding
targets. This guard looks at a whole chunk before any row of it is sent and refuses chunks that look wrong,
which the 300 Hz thread would otherwise only slow down:

    non-finite value                               -> reject
    |row 0 - measured state| on an arm > 0.3 rad   -> reject (misread state, wrong checkpoint, wrong layout)
    arm step between consecutive rows > 0.15 rad  -> reject (above the largest step in the demonstrations)
    arm step > 0.07 rad                            -> counted only (above the demonstrations' p99)

Thresholds come from the 208k-frame T-Rex tong-transfer demos (CHOICES.md): arm |action - state| p99 0.28 /
max 0.56 rad; arm step between consecutive actions p99 0.065 / max 0.135 rad per frame.
`filter_row` additionally clips to the joint limits (when given) so the counts show how often the policy
asks for an out-of-range joint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ARM = np.r_[0:7, 29:36]


@dataclass
class GuardLimits:
    row0_max: float = 0.3
    step_count: float = 0.07
    step_reject: float = 0.15


@dataclass
class GuardCounts:
    chunks: int = 0
    rejected: int = 0
    step_hits: int = 0
    limit_hits: int = 0
    max_row0: float = 0.0
    max_step: float = 0.0
    reasons: list[str] = field(default_factory=list)


class SharpaGuard:
    def __init__(self, limits: GuardLimits | None = None,
                 joint_limits: tuple[np.ndarray, np.ndarray] | None = None):
        self.limits = limits or GuardLimits()
        self.joint_limits = joint_limits
        self.cumulative = GuardCounts()

    def check_chunk(self, chunk: np.ndarray, state: np.ndarray, rows: int) -> str | None:
        """Look at the rows that will be executed. Returns None if OK, else the reason to reject."""
        c = self.cumulative
        c.chunks += 1
        part = np.asarray(chunk, dtype=np.float64)[:rows]
        reason = None
        if not np.all(np.isfinite(part)):
            reason = "non-finite value in chunk"
        else:
            row0 = float(np.abs(part[0, ARM] - np.asarray(state)[ARM]).max())
            steps = np.abs(np.diff(part[:, ARM], axis=0)).max(axis=1) if rows > 1 else np.zeros(0)
            c.max_row0 = max(c.max_row0, row0)
            c.step_hits += int((steps > self.limits.step_count).sum())
            if steps.size:
                c.max_step = max(c.max_step, float(steps.max()))
            if row0 > self.limits.row0_max:
                reason = f"row 0 is {row0:.3f} rad from the measured arm state (limit {self.limits.row0_max})"
            elif steps.size and steps.max() > self.limits.step_reject:
                i = int(steps.argmax())
                reason = f"arm step {steps[i]:.3f} rad between rows {i} and {i + 1} (limit {self.limits.step_reject})"
        if reason:
            c.rejected += 1
            c.reasons.append(reason)
        return reason

    def filter_row(self, row: np.ndarray) -> tuple[np.ndarray, bool]:
        """Clip to the joint limits (if known). Returns (row, clipped)."""
        if self.joint_limits is None:
            return row, False
        lo, hi = self.joint_limits
        out = np.clip(row, lo, hi)
        clipped = bool(np.any(out != row))
        self.cumulative.limit_hits += int(clipped)
        return out, clipped

    def summary(self) -> dict:
        c = self.cumulative
        return {"chunks": c.chunks, "rejected": c.rejected, "step_hits": c.step_hits, "limit_hits": c.limit_hits,
                "max_row0_rad": round(c.max_row0, 4), "max_step_rad": round(c.max_step, 4),
                "reasons": c.reasons[-10:]}
