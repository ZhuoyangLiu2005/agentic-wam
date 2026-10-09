"""Ego (human egocentric video) raw-76D <-> unified-64D delta-EEF transform.

Unified 64-D = 2 x [ 9-D EEF (pos3 + rot6d) | 22-D hand | 1-D gripper ]  (Left block, then Right).
Human hands are retargeted to a 22-joint dexterous hand, so the hand block is POPULATED and the
gripper slot is ZERO -- the mirror of YAM (gripper live, hand zero).

RAW 76-D LAYOUT of an ego dataset (docs/PRETRAINING.md §1.2):
    [ 0:16]  L wrist pose, 4x4 homogeneous, ROW-MAJOR flattened, world frame, metres
    [16:32]  R wrist pose, 4x4 homogeneous, ROW-MAJOR flattened
    [32:54]  L hand, 22 joint angles (radians), retargeted
    [54:76]  R hand, 22 joint angles (radians), retargeted
`observation.state` is the pose at frame t and `action` the target the hands reach next (for
recorded human motion, the state at t+1). Both absolute.

This is the same layout as the Tianji + Sharpa robot (arms + 22-DoF hands), so the math is shared:
STATE = absolute EEF + absolute hand; ACTION = EEF delta in the CHUNK-START frame + absolute hand.
"""
from __future__ import annotations

from .tianji_sharpa_unified_eef import RAW_DIM, TianjiSharpaUnifiedEEFTransform  # noqa: F401


class EgoUnifiedEEFTransform(TianjiSharpaUnifiedEEFTransform):
    """raw 76-D ego -> unified 64-D. Plugs into AgenticWAMProcessor.action_state_transforms."""
