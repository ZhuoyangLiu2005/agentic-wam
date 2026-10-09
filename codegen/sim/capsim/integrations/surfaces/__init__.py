"""Declarative API surfaces: names, signatures and docstrings only (no robot/vision imports), so a prompt
renders anywhere via ``ArmApi.__new__(ArmApi).combined_doc()``. Backends override bodies only and inherit
the docstrings, so the prompt cannot drift. :class:`ArmApi` = motion, perception, grasp synthesis;
:class:`GripperApi` adds the parallel-jaw primitives.
"""

from capsim.integrations.surfaces.arm import ArmApi
from capsim.integrations.surfaces.gripper import GripperApi

__all__ = ["ArmApi", "GripperApi"]
