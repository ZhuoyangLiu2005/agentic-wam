"""Grounding (sensor side): sub_task slicing, gripper-transition detection,
and per-slice observed events."""
from .facts import block_events
from .segment import smooth, sub_task_blocks
from .transitions import grip_transitions

__all__ = ["block_events", "grip_transitions", "smooth", "sub_task_blocks"]
