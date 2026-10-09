"""ArmApi — the declarative end-effector-agnostic surface: arm motion, perception/grounding, grasp synthesis.

Bodies raise NotImplementedError; backends (``robotwin/base.py``) implement them. The docstrings and signatures
here ARE the codegen prompt: edit prompt text only in this file. They document RoboTwin's camera rig and frames,
which generated programs rely on; the arm selector is ``hand`` ("left"/"right") on every surface.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from capsim.integrations.base_api import ApiBase


class ArmApi(ApiBase):
    """Declarative arm + perception surface (see module docstring)."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Doc-less overrides inherit the ancestor docstring, so a backend can never change the prompt.
        super().__init_subclass__(**kwargs)
        for name, member in vars(cls).items():
            if inspect.isfunction(member) and member.__doc__ is None:
                for parent in cls.__mro__[1:]:
                    doc = getattr(getattr(parent, name, None), "__doc__", None)
                    if doc is not None:
                        member.__doc__ = doc
                        break

    def functions(self) -> dict[str, Any]:
        # Order is part of the prompt; GripperApi re-declares the full registry.
        return {
            "get_observation": self.get_observation,
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "get_oriented_bounding_box_from_3d_points": self.get_oriented_bounding_box_from_3d_points,
            "get_object_3d_points_and_masks_from_language": self.get_object_3d_points_and_masks_from_language,
            "goto_home_joint_position": self.goto_home_joint_position,
        }

    def get_observation(self) -> dict[str, Any]:
        """Get the observation of the environment.
        Returns:
            observation:
                A dictionary containing the observation of the environment.
                Four cameras are available: "head_camera" (overhead scene view),
                "front_camera" (front scene view), "left_camera" (left-wrist,
                moves with the left arm) and "right_camera" (right-wrist, moves
                with the right arm). Each camera key maps to the same structure:
                - [<camera>]["images"]["rgb"]: Current color image as a numpy array of shape (H, W, 3), dtype uint8.
                - [<camera>]["images"]["depth"]: Current depth image as a numpy array of shape (H, W), dtype float32, in meters (0 marks invalid pixels).
                - [<camera>]["intrinsics"]: Camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - [<camera>]["pose_mat"]: Camera extrinsic matrix (camera -> world) as a numpy array of shape (4, 4), dtype float64.
                Robot state keys:
                - ["left_endpose"]: Current left end-effector pose in the world frame as a numpy array of shape (7,), dtype float64. The first 3 elements are the end-effector XYZ, the next 4 elements are the quaternion wxyz.
                - ["right_endpose"]: Current right end-effector pose in the world frame as a numpy array of shape (7,), dtype float64. Same layout as left_endpose.
                - ["robot_joint_pos"]: Current joint positions of both arms as a numpy array, dtype float64 (left arm joints followed by right arm joints).
        """
        raise NotImplementedError("declarative surface - attach a robot backend")

    def get_object_pose(self, object_name: str, use_multiview: bool = True) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Get the pose of an object in the environment from a natural language description.
        The quaternion from get_object_pose may be unreliable, so disregard it and use the grasp
        pose quaternion OR (0.5, -0.5, 0.5, 0.5) wxyz as the gripper down orientation if using this for
        placement position.
        It is possible that get_object_pose is sometimes unreliable and will return None for both
        position and quaternion.

        Args:
            object_name: The name of the object to get the pose of, in underscore separated lowercase words.
            use_multiview: If True, uses every available camera (front and both wrist cameras) as well as the head camera for segmentation. If False, only uses the head camera.

        Returns:
            position: (3,) XYZ in meters (world frame). Origin convention: the object's BASE
                (its contact point with the support surface) for objects that stand or lie on the
                table; the CENTRE for blocks/cubes, boxes and pots; the base END for a bottle lying
                on its side. Every backend follows the same convention, so heights computed from it
                (rest height, release height) transfer between backends.
            quaternion_wxyz: (4,) WXYZ unit quaternion (world frame). Only its yaw about world z is
                meaningful; use it as a top-down wrist orientation, not as the object's full heading.
        """
        raise NotImplementedError("declarative surface - attach a robot backend")

    def sample_grasp_pose(
        self,
        object_name: str,
        use_multiview: bool = True,
        hand: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        Uses multiview point clouds and plan_grasp_from_point_clouds for more reliable grasp planning.
        NOTE: This function is best for generating grasps on objects with complex shapes. For regular objects like boxes, cylinders, etc..., use get_top_down_grasp_from_obb.
        Returns the highest-scoring grasp candidate that the motion planner can currently reach.
        Raises ValueError when the object cannot be grounded or no reachable grasp exists;
        candidates are stochastic, so calling again can succeed - wrap in try/except and retry.
        Do use the grasp sample quaternion from sample_grasp_pose.

        Args:
            object_name: The name of the object to sample a grasp pose for, in underscore separated lowercase words.
            use_multiview: If True, uses every available camera (front and both wrist cameras) as well as the head camera for segmentation. If False, only uses the head camera.
            hand: "left" or "right" - pass the arm you intend to grasp with so the returned grasp is validated as reachable for THAT arm. Strongly recommended whenever you know the arm (e.g. re-grasping in the shared band). Default None validates against either arm.

        Returns:
            position: (3,) XYZ in meters (world frame).
            quaternion_wxyz: (4,) WXYZ unit quaternion (world frame).
        """
        raise NotImplementedError("declarative surface - attach a robot backend")

    def goto_pose(
        self, hand: str, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> bool:
        """Go to pose using motion planning.
        There is no need to call a second goto_pose with the same position and quaternion_wxyz after calling it with z_approach.
        Args:
            hand: "left" or "right" — which arm to move.
            position: (3,) XYZ in meters (world frame).
            quaternion_wxyz: (4,) WXYZ unit quaternion (world frame).
            z_approach: (float) Approach-axis distance offset for goto_pose insertion approach motion. Will first arrive at a pose backed off z_approach meters along the gripper's approach axis before moving to the requested pose. Useful for more precise grasp approaches. Default is 0.0.
        Returns:
            True on success, False if motion planning failed.
        """
        raise NotImplementedError("declarative surface - attach a robot backend")

    def get_oriented_bounding_box_from_3d_points(self, points: np.ndarray) -> dict[str, Any]:
        """Get the oriented bounding box from 3D points.

        Args:
            points: np.ndarray: The 3D points to get the oriented bounding box from.
                Shape: (N, 3), dtype float64.

        Returns:
            dict[str, Any]: The oriented bounding box. The dictionary contains the following keys:
                - "center": np.ndarray: The center of the oriented bounding box in point cloud frame.
                - "extent": np.ndarray: The extent of the oriented bounding box.
                - "R": np.ndarray: The rotation matrix of the oriented bounding box in point cloud frame.

        Example:
            >>> points = np.random.randn((100, 3))
            >>> obb = get_oriented_bounding_box_from_3d_points(points)
        """
        raise NotImplementedError("declarative surface - attach a robot backend")

    def get_object_3d_points_and_masks_from_language(
        self,
        text_prompt: str,
        use_multiview: bool = True,
    ) -> dict[str, Any]:
        """Segment an object using text prompt across multiple camera views.

        Uses Molmo to locate the object in each camera view, then SAM3 for
        segmentation. Views whose 3D points agree with the head view (within
        1 cm) are fused; a view that cannot see the object is skipped.

        Raises ValueError when the description grounds to an implausible region
        (false detection) - wrap in try/except and retry with a different description.

        Args:
            text_prompt: Text description of the object to segment.
            use_multiview: If True, uses every available camera (front and both
                wrist cameras) in addition to the head camera. If False, only
                the head camera is used.

        Returns:
            dict containing:
                - "head_mask": np.ndarray of shape (H, W), dtype bool
                - "head_points_3d": np.ndarray of shape (N, 3), head-view points in world frame
                - "head_score": float, SAM3 confidence score of the head view
                - "points_3d": np.ndarray of shape (N, 3), fused 3D points in world frame
                - "camera_masks": dict camera name -> mask for every view where segmentation succeeded
                - "camera_scores": dict camera name -> SAM3 confidence score
                - "camera_points_3d": dict camera name -> (N, 3) world-frame points of that view
        """
        raise NotImplementedError("declarative surface - attach a robot backend")

    def goto_home_joint_position(self, hand: str) -> bool:
        """Return the arm to its reset joint configuration with high manipulability.

        Args:
            hand: "left" or "right" — which arm to return home.
        """
        raise NotImplementedError("declarative surface - attach a robot backend")


__all__ = ["ArmApi"]
