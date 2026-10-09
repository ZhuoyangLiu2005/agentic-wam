"""GripperApi — :class:`ArmApi` plus the parallel-jaw primitives; its ``combined_doc`` is the codegen prompt.

``functions()`` fixes the canonical prompt order (gripper functions interleaved; order is part of the prompt).
Backend: :class:`capsim.integrations.robotwin.gripper.RoboTwinGripperApi`, never privileged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from capsim.integrations.surfaces.arm import ArmApi


class GripperApi(ArmApi):
    """Declarative ArmApi + parallel-jaw primitives (see module docstring)."""

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "get_oriented_bounding_box_from_3d_points": self.get_oriented_bounding_box_from_3d_points,
            "get_top_down_grasp_from_obb": self.get_top_down_grasp_from_obb,
            "get_object_3d_points_and_masks_from_language": self.get_object_3d_points_and_masks_from_language,
            "goto_home_joint_position": self.goto_home_joint_position,
            "get_gripper_aperture": self.get_gripper_aperture,
        }

    def open_gripper(self, hand: str) -> bool:
        """Open gripper fully.

        Args:
            hand: "left" or "right" — which arm's gripper to open.
        """
        raise NotImplementedError("declarative surface - attach a robot backend")

    def close_gripper(self, hand: str) -> bool:
        """Close gripper fully.

        Args:
            hand: "left" or "right" — which arm's gripper to close.
        """
        raise NotImplementedError("declarative surface - attach a robot backend")

    def get_gripper_aperture(self, hand: str) -> float:
        """MEASURED normalized gripper opening: 1.0 = fully open, 0.0 = the jaws closed on nothing.

        Read from the finger joint's actual position (not the commanded target) and
        calibrated so that closing on air reads 0, so an object physically blocking the
        fingers shows up as a plateau above 0 after close_gripper — like a real robot's
        gripper readback (a 3 cm block reads about 0.4, a 5 cm block about 0.7). Use it
        to VERIFY a grasp: a value below ~0.05 right after close_gripper means the jaws
        closed on air (re-perceive and try another grasp); the plateau value scales with
        the held object's width, and a drop toward 0 during transport means the object
        was lost.

        Args:
            hand: "left" or "right" — which arm's gripper to read.

        Returns:
            Normalized measured opening in [0, 1].
        """
        raise NotImplementedError("declarative surface - attach a robot backend")

    def get_top_down_grasp_from_obb(
        self,
        obb: dict,
        hand: str | None = None,
        points: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute a straight top-down grasp pose from an oriented bounding box.
        NOTE: This is the most reliable way to grasp regular objects like boxes and
        cylinders standing on the table: the gripper descends vertically onto the
        object's centre with the jaws closing across its narrowest horizontal width.
        For objects with complex shapes use sample_grasp_pose instead.

        Args:
            obb: The oriented bounding box dict returned by
                get_oriented_bounding_box_from_3d_points (keys "center", "extent", "R").
            hand: "left" or "right" - pass the arm you intend to grasp with so the
                pose is validated as reachable for that arm. Default None validates
                against either arm.
            points: STRONGLY RECOMMENDED - the object's (N, 3) world-frame points
                (e.g. "head_points_3d"). When given, the grasp centre and jaw yaw are
                estimated robustly from the object's top face instead of the noisier
                3D bounding box.

        Returns:
            position: (3,) XYZ in meters (world frame) - pass straight to goto_pose
                (use z_approach for a clean vertical descent).
            quaternion_wxyz: (4,) WXYZ unit quaternion (world frame).
        """
        raise NotImplementedError("declarative surface - attach a robot backend")


__all__ = ["GripperApi"]
