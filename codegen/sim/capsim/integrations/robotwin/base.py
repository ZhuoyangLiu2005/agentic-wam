"""RoboTwin (dual-arm ALOHA) backend of the declarative :class:`ArmApi` surface.

Perception (Molmo points + SAM3 masks, fused over every enabled camera), grasp synthesis (Contact-GraspNet)
and motion (RoboTwin's planner). Prompt-facing docstrings live on the surface (``surfaces/arm.py``) and are
inherited, so a method here without a docstring cannot change the codegen prompt.

Conventions: depth is metres with 0 = invalid (hence ``depth > 0`` and ``filter_invalid=False`` wherever points
are indexed per pixel); motion methods take ``hand`` ("left"/"right") and return True/False because planning
can fail; ``position`` in ``goto_pose`` is the fingertip point; RoboTwin's endpose approach axis is tool +x
(jaws close along y), so z-approach frames (GraspNet, OBB) are converted with ``_ZAPPROACH_TO_TCP_WXYZ``.
Requires the SAM3, Molmo and Contact-GraspNet servers (YAML ``api_servers``).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

import numpy as np
import open3d as o3d
import viser.transforms as vtf
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SciRotation
from sklearn.cluster import DBSCAN

from capsim.integrations.surfaces.arm import ArmApi
from capsim.integrations.vision.graspnet import init_contact_graspnet_point_clouds
from capsim.integrations.vision.molmo import init_molmo, init_molmo_all_points
from capsim.integrations.vision.sam3 import init_sam3, init_sam3_point_prompt
from capsim.integrations.robotwin import vocab as _vocab
from capsim.integrations.robotwin.code_alignment import record_motion
from capsim.profiles import embodiment_of, load_profile
from capsim.utils.depth_utils import depth_to_pointcloud


class RoboTwinBaseApi(ArmApi):
    """RoboTwin (dual-arm ALOHA) backend of ArmApi, FrankaLiberoApi design."""

    # Embodiment constants (_TCP_OFFSET, _FINGERTIP_OFFSET_M, _ZAPPROACH_TO_TCP_WXYZ, table, cameras, aperture
    # calibration) are bound per instance from capsim/profiles/<embodiment>.yaml by _bind_profile.
    # --- tuning values: grounding_policy.yaml (units and meaning documented there) ---
    _P = _vocab.policy()
    _GOTO_MISS_LIMIT = float(_P["motion"]["goto_miss_limit"])
    _WS_EDGE_MARGIN = _P["workspace"]["edge_margin"]
    _WS_BELOW_SURFACE = _P["workspace"]["below_surface"]
    _WS_ABOVE_SURFACE = _P["workspace"]["above_surface"]
    _MAX_OBJECT_EXTENT = _P["workspace"]["max_object_extent"]
    _VIEW_CONSISTENCY_M = _P["fusion"]["view_consistency"]
    _POINT_MERGE_M = _P["fusion"]["point_merge"]
    _SEG_MIN_SCORE = _P["segmentation"]["min_score"]   # below it: phantom mask, the grounding fails instead of a wrong pose

    def _bind_profile(self, env: Any) -> None:
        """Bind the embodiment profile the env names (``RoboTwinEnv.embodiment``) to this instance."""
        prof = load_profile(embodiment_of(env))
        self._profile = prof
        self._TCP_OFFSET = np.array(prof.tool.tcp_offset, dtype=np.float64)
        self._FINGERTIP_OFFSET_M = float(prof.tool.fingertip_offset_m)
        self._ZAPPROACH_TO_TCP_WXYZ = np.array(prof.tool.zapproach_to_tcp_wxyz, dtype=np.float64)
        self._TABLE_HALF_XY = tuple(float(v) for v in prof.table.half_xy)
        self._ROBOT_DIR = np.array(prof.robot.base_dir, dtype=np.float64)
        self._APERTURE_AIR_RAW = float(prof.gripper.aperture_air_raw)
        self._APERTURE_EMPTY = float(prof.gripper.aperture_empty)
        self.camera_name = str(prof.cameras.primary)
        # Which aux cameras exist depends on the task config: multiview code checks membership in obs first.
        self.aux_camera_names = [str(c) for c in prof.cameras.aux]
        self._front_camera = str(prof.cameras.front)
        self._wrist_camera = {str(k): str(v) for k, v in prof.cameras.wrist.items()}

    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self._rt_env = env
        self._bind_profile(env)

        self.sam3_seg_fn = init_sam3()
        self.sam3_point_prompt_fn = init_sam3_point_prompt()
        self.molmo_point_fn = init_molmo()
        self.molmo_all_points_fn = init_molmo_all_points()
        self.grasp_net_plan_point_clouds_fn = init_contact_graspnet_point_clouds()

    @property
    def task(self):
        return self._rt_env.task

    def _workspace_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Tabletop AABB (world frame) from the live table pose, so randomized table height/xy are tracked."""
        table_pos = np.asarray(self.task.table.get_pose().p, dtype=np.float64)  # TODO(profile): sim read on the perception tier
        hx, hy = self._TABLE_HALF_XY
        m = self._WS_EDGE_MARGIN
        lo = np.array(
            [table_pos[0] - hx - m, table_pos[1] - hy - m, table_pos[2] - self._WS_BELOW_SURFACE]
        )
        hi = np.array(
            [table_pos[0] + hx + m, table_pos[1] + hy + m, table_pos[2] + self._WS_ABOVE_SURFACE]
        )
        return lo, hi

    def _arm(self, hand: str):
        from envs.utils.action import ArmTag  # type: ignore[import-not-found]

        return ArmTag(hand)

    def _arm_joints(self, hand: str) -> np.ndarray:
        """Current measured arm joint angles (gripper dropped), shape (6,)."""
        js = (
            self.task.robot.get_left_arm_real_jointState()
            if hand == "left"
            else self.task.robot.get_right_arm_real_jointState()
        )
        return np.asarray(js[:6], dtype=np.float64)

    def functions(self) -> dict[str, Any]:
        return ArmApi.functions(self)

    def get_observation(self) -> dict[str, Any]:
        obs = self._rt_env.get_observation()
        for cam_name in [self.camera_name, *self.aux_camera_names]:
            if cam_name not in obs:
                continue
            depth = obs[cam_name]["images"]["depth"]
            if depth.ndim == 3:
                obs[cam_name]["images"]["depth"] = depth.squeeze(-1)
        return obs

    def segment_sam3_point_prompt(
        self,
        rgb: np.ndarray,
        point_coords: tuple[float, float],
    ) -> list[dict[str, Any]]:
        """SAM3 masks for the instance at pixel ``point_coords`` (x, y) of an (H, W, 3) uint8 image.

        Returns:
            A list of dicts with "mask" ((H, W) bool) and "score" (float).
        """
        return self.sam3_point_prompt_fn(Image.fromarray(rgb), point_coords)

    def segment_sam3_text_prompt(
        self,
        rgb: np.ndarray,
        text_prompt: str,
    ) -> list[dict[str, Any]]:
        """SAM3 masks for ``text_prompt`` on an (H, W, 3) uint8 image.

        Returns:
            A list of dicts with "mask" ((H, W) bool), "box" ([x1, y1, x2, y2] px) and "score"; [] if none.
        """
        results = self.sam3_seg_fn(rgb, text_prompt=text_prompt)
        if len(results) == 0:
            print(f"[segment_sam3_text_prompt] SAM3 returned no results for prompt: '{text_prompt}'")
            return []
        return results

    def point_prompt_molmo(
        self,
        image: np.ndarray,
        text_prompt: str,
    ) -> dict[str, tuple[int | None, int | None]]:
        """Molmo pixel point for ``text_prompt`` on an (H, W, 3) uint8 image.

        Returns:
            {query: (x, y)} pixel coordinates; (None, None) if parsing failed.
        """
        return self.molmo_point_fn(Image.fromarray(image), objects=[text_prompt])

    def _pose7(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> list:
        p = np.asarray(position, dtype=np.float64).reshape(3)
        q = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        return [*p.tolist(), *q.tolist()]

    def _exec(self, *plans: Any) -> bool:
        plans = tuple(pl for pl in plans if pl is not None)
        if not plans:
            return False
        # Clear RoboTwin's episode-wide plan_success latch so each call is judged on its own plan; the planner is
        # stochastic and a failed plan moves nothing, so one retry is side-effect-free.
        frame_start = int(getattr(self.task, "FRAME_IDX", 0) or 0)
        for _attempt in range(2):
            self.task.plan_success = True
            result = self.task.move(plans[0]) if len(plans) == 1 else self.task.move(plans[0], plans[1])
            ok = bool(result) if result is not None else True
            if ok:
                break
        self._rt_env._sim_step_count += 1
        if getattr(self.task, "save_data", False):
            frame_end = int(getattr(self.task, "FRAME_IDX", 0) or 0)
            if frame_end > frame_start:
                caller = sys._getframe(1)          # the API method that invoked _exec, for frame attribution
                record_motion(
                    self._rt_env, caller.f_code.co_name, frame_start, frame_end,
                    hand=caller.f_locals.get("hand"),
                )
        return ok

    def goto_pose(
        self, hand: str, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0
    ) -> bool:
        arm = self._arm(hand)
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        offset_pos = pos + rot.apply(self._TCP_OFFSET)  # fingertip -> planner endpose, in the tool frame

        approach_ok: bool | None = None
        if (
            z_approach != 0.0
        ):  # back off along the approach axis (tool -x) first
            z_offset_pos = offset_pos + rot.apply(np.array([-z_approach, 0, 0]))
            approach_ok = self._exec(self.task.move_to_pose(arm, self._pose7(z_offset_pos, quat_wxyz)))

        final_ok: bool | None = None
        if approach_ok is not False:
            final_ok = self._exec(self.task.move_to_pose(arm, self._pose7(offset_pos, quat_wxyz)))

        # A True plan result does not mean the arm arrived: log the fingertip miss.
        if final_ok:
            try:
                _, tip = self._fingertip_of(hand)
                _miss_limit = self._GOTO_MISS_LIMIT
                d = float(np.linalg.norm(np.asarray(tip, dtype=np.float64) - pos))
                q_now = np.asarray(self.task.get_arm_pose(arm), dtype=np.float64)[3:7]
                dq = SciRotation.from_quat([q_now[1], q_now[2], q_now[3], q_now[0]]).inv() * rot
                ang = float(np.degrees(np.linalg.norm(dq.as_rotvec())))
                if d > 0.005 or ang > 5.0:
                    print(f"[goto] {hand} commanded {np.round(pos, 4).tolist()} achieved "
                          f"{np.round(np.asarray(tip, dtype=np.float64), 4).tolist()} miss {d * 100:.1f} cm "
                          f"{ang:.0f} deg")
                if d > _miss_limit:
                    print(f"[goto] {hand} MISSED by {d * 100:.1f} cm (limit {_miss_limit * 100:.0f}) -> reporting failure")
                    return False
            except Exception as _exc:                     # observability must never change control flow
                print(f"[goto] achieved-pose read failed: {str(_exc)[:60]}")

        return bool(final_ok)

    # Per-object origin conventions, keyed by the words of the name (object_vocab.yaml `origin`).
    _ORIGIN = _vocab.section("origin")
    _BLOCK_RE = _vocab.words_re(_ORIGIN["centre_words"])
    _CENTRE_RE = _vocab.names_re(_ORIGIN["centre_names"]["names"], _ORIGIN["centre_names"]["prefixes"])
    _BOTTLE_RE = _vocab.words_re(_ORIGIN["base_end_words"])
    _LYING_RE = _vocab.words_re(_ORIGIN["lying_cylinder_words"])
    _FLOOR_RE = _vocab.words_re(_ORIGIN["floor_words"])
    _TABLETOP_RE = _vocab.words_re(_ORIGIN["tabletop_words"])

    def _is_floor_name(self, name) -> bool:
        """True for a name denoting a floor-standing object ("table trash bin" is not one)."""
        n = name or ""
        return bool(self._FLOOR_RE.search(n)) and not self._TABLETOP_RE.search(n)

    def _apply_origin_convention(self, name, position, pts, table_z, z_top, z_bot, resting):
        """Map a perceived cloud onto the privileged tier's actor-origin convention (base / centre / base end)."""
        pos = np.array(position, dtype=np.float64)
        # Large objects: the visible centroid sits on the near wall; use the top-face centre.
        try:
            ext = np.percentile(pts[:, :2], 98, axis=0) - np.percentile(pts[:, :2], 2, axis=0)
            if float(np.max(ext)) > 0.15 and resting:
                c_top = self._top_face_centre(pts)
                if c_top is not None and float(np.linalg.norm(c_top - pos[:2])) < 0.15:
                    pos[0], pos[1] = float(c_top[0]), float(c_top[1])
        except Exception:
            pass
        # Lifted: fingers hide the bottom, so use the height remembered while it rested.
        h_rest = getattr(self, "_rest_height", {}).get(name)
        lifted_base = (z_top - h_rest) if (not resting and h_rest is not None and h_rest > 0.005) else z_bot
        if self._BLOCK_RE.search(name or "") or self._CENTRE_RE.match((name or "").strip()):
            pos[2] = min(pos[2], (z_top + table_z) / 2.0) if resting else (z_top + lifted_base) / 2.0
            return pos
        if self._is_floor_name(name) and z_bot < table_z - 0.05:
            pos[2] = z_bot
            return pos
        if not self._LYING_RE.search(name or ""):
            pos[2] = table_z if resting else lifted_base
            return pos
        try:
            xy = pts[:, :2] - pts[:, :2].mean(axis=0)
            if len(xy) > 3000:
                xy = xy[:: (len(xy) // 3000) + 1]
            cov = np.cov(xy.T)
            evals, evecs = np.linalg.eigh(cov)
            axis = evecs[:, int(np.argmax(evals))]
            proj = xy @ axis
            perp = xy @ np.array([-axis[1], axis[0]])
            L = float(np.percentile(proj, 98) - np.percentile(proj, 2))
            W = float(np.percentile(perp, 98) - np.percentile(perp, 2))
            H = z_top - (table_z if resting else z_bot)
        except Exception:
            L, W, H = 0.0, 1.0, 0.0
        if L > 2.5 * W and H <= 1.3 * W + 0.01 and W > 0.02:
            if not self._BOTTLE_RE.search(name or ""):
                # non-bottle: origin on the axis centre, one radius above the table
                pos[2] = (table_z + W / 2.0) if resting else (z_top + z_bot) / 2.0
                return pos
            # the base end is the half with the larger perpendicular extent
            lo_half = perp[proj < np.median(proj)]
            hi_half = perp[proj >= np.median(proj)]
            w_lo = float(np.percentile(lo_half, 95) - np.percentile(lo_half, 5)) if len(lo_half) > 10 else 0.0
            w_hi = float(np.percentile(hi_half, 95) - np.percentile(hi_half, 5)) if len(hi_half) > 10 else 0.0
            end = float(np.percentile(proj, 2)) if w_lo >= w_hi else float(np.percentile(proj, 98))
            base_xy = pts[:, :2].mean(axis=0) + end * axis
            pos[0], pos[1] = float(base_xy[0]), float(base_xy[1])
            pos[2] = (table_z + W / 2.0) if resting else (z_top + lifted_base) / 2.0
            print(f"[origin] '{name}' lying elongated (L {L:.3f} W {W:.3f}) -> base end")
            return pos
        pos[2] = table_z if resting else lifted_base
        return pos

    def get_object_pose(self, object_name: str, use_multiview: bool = True) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        start_time = time.time()

        # Contract: a failed grounding returns (None, None) instead of raising.
        try:
            result = self.get_object_3d_points_and_masks_from_language(
                object_name, use_multiview=use_multiview
            )
        except ValueError as exc:
            print(f"get_object_pose: '{object_name}' not grounded ({exc})")
            if not re.search(r"not grounded|not confidently|not currently visible|size-superlative|returned no|not found|No grasp|instances visible", str(exc)):
                import traceback
                traceback.print_exc()          # an internal error, not the grounding contract
            return None, None
        points_3d = result["points_3d"]

        if len(points_3d) == 0:
            print(f"get_object_pose: '{object_name}' not grounded (empty cloud)")
            return None, None

        raw_points = np.asarray(points_3d)
        points_3d, _ = self.filter_noise(points_3d)
        if len(points_3d) < 10 and len(raw_points) >= 10:
            print(f"get_object_pose: '{object_name}' cleaner left {len(points_3d)} of {len(raw_points)} points -> using the raw cloud")
            points_3d = raw_points
        if len(points_3d) == 0:
            print(f"get_object_pose: '{object_name}' not grounded (no points after cleaning)")
            return None, None

        obb = self.get_oriented_bounding_box_from_3d_points(points_3d)

        position = np.array(obb["center"])
        R = np.array(obb["R"])

        # Parity with the privileged tier: pure top-down + yaw (partial-cloud OBB axes are tilted). Yaw from the
        # top-face min-area rectangle, else the most-horizontal OBB axis.

        # --- Perception-backend contract parity with the privileged backend ---
        yaw = None
        pts_arr = np.asarray(points_3d)
        z_top_for_yaw = float(np.percentile(pts_arr[:, 2], 95))
        top = pts_arr[pts_arr[:, 2] > z_top_for_yaw - 0.01]
        if len(top) >= 30:
            yaw = self._min_area_rect_angle(top[:, :2])
        if yaw is None:
            horiz = min(range(3), key=lambda i: abs(R[2, i]))
            xy = R[:2, horiz]
            yaw = (float(np.arctan2(xy[1], xy[0]))
                   if np.linalg.norm(xy) > 1e-6 else 0.0)
        if os.environ.get("CAPX_YAW_DEBUG"):
            print(f"[yawdbg] '{object_name}' n_pts={len(pts_arr)} "
                  f"n_top={len(top)} z_top={z_top_for_yaw:.3f} "
                  f"yaw_deg={np.degrees(yaw):.1f}")
        x_ax = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        z_ax = np.array([0.0, 0.0, -1.0])
        R = np.stack([x_ax, np.cross(z_ax, x_ax), z_ax], axis=1)
        # The OBB z-centre is ~h/2 too high (only the top face is seen): map onto the origin convention.
        z_top = float(np.percentile(np.asarray(points_3d)[:, 2], 95))
        z_bot = float(np.percentile(np.asarray(points_3d)[:, 2], 5))
        table_z = self._table_plane_z()
        if table_z is not None and z_top > table_z:
            gap, vis_h = z_bot - table_z, z_top - z_bot
            held_near = False
            sub = pts_arr[:: max(1, len(pts_arr) // 2000)]
            for h_, (pose_, tip_, closed_) in self._all_fingertips().items():
                if closed_ and len(sub):
                    # distance to the cloud, not the centre: handles are far from the centre
                    held_near = held_near or float(np.min(np.linalg.norm(sub - tip_, axis=1))) < 0.06
            explicitly_held = (self._holder_of(object_name) is not None
                               or getattr(self, "_last_served_inhand", None) == object_name)
            # Off the table only near a closed gripper: a top-only mask of a standing can has a large gap but rests.
            resting = (not explicitly_held) and (gap < 0.02 or not held_near)
            if not hasattr(self, "_rest_height"):
                self._rest_height = {}
            if resting:
                self._rest_height[object_name] = z_top - table_z
            position = self._apply_origin_convention(object_name, position, pts_arr, table_z,
                                                     z_top, z_bot, resting)

        quaternion_wxyz = (
            vtf.SO3.from_matrix(R) @ vtf.SO3(wxyz=self._ZAPPROACH_TO_TCP_WXYZ)
        ).wxyz

        print(f"get_object_pose in {time.time() - start_time} seconds")
        return position, quaternion_wxyz

    _FEASIBILITY_TOP_K = _P["grasp"]["feasibility_top_k"]
    _PROBE_BACKOFF = _P["grasp"]["probe_backoff"]

    def _grasp_plan_feasible(
        self,
        tcp_position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        hand: str | None = None,
    ) -> bool:
        """Plan-only probe (nothing moves): True if ``hand`` (or, when None, either arm) plans both grasp legs."""
        pos = np.asarray(tcp_position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64
        )
        rot = SciRotation.from_quat(quat_xyzw)
        endpose_pos = pos + rot.apply(self._TCP_OFFSET)
        approach_pos = endpose_pos - self._PROBE_BACKOFF * rot.apply(np.array([1.0, 0.0, 0.0]))
        if hand == "left":
            plan_fns = [self.task.robot.left_plan_path]
        elif hand == "right":
            plan_fns = [self.task.robot.right_plan_path]
        else:
            plan_fns = [self.task.robot.left_plan_path, self.task.robot.right_plan_path]
            if self._profile.near_hand(pos) == "right":
                plan_fns.reverse()
        for plan_fn in plan_fns:
            try:
                ok = True
                for target in (approach_pos, endpose_pos):
                    result = plan_fn([*target.tolist(), *quat_wxyz.tolist()])
                    if not (bool(result) and result.get("status") == "Success"):
                        ok = False
                        break
                if ok:
                    return True
            except Exception:
                continue
        return False

    def sample_grasp_pose(
        self,
        object_name: str,
        use_multiview: bool = True,
        hand: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        result = self.get_object_3d_points_and_masks_from_language(
            object_name, use_multiview=use_multiview
        )
        pc_segment = result["points_3d"]

        if len(pc_segment) == 0:
            raise ValueError(f"Could not segment object '{object_name}'")

        obs = self.get_observation()
        pc_full_parts = []
        for cam_name in [self.camera_name, *self.aux_camera_names]:
            if cam_name not in obs:
                continue
            depth = obs[cam_name]["images"]["depth"]
            intrinsics = obs[cam_name]["intrinsics"]
            extrinsics = obs[cam_name]["pose_mat"]
            pts_camera = depth_to_pointcloud(depth, intrinsics, subsample_factor=1)
            pts_homogeneous = np.concatenate(
                [pts_camera, np.ones((len(pts_camera), 1))], axis=1
            )
            pts_world = (extrinsics @ pts_homogeneous.T).T[:, :3]
            pc_full_parts.append(pts_world)
        pc_full = np.concatenate(pc_full_parts)

        pc_segment, _ = self.filter_noise(pc_segment)
        if len(pc_segment) == 0:
            raise ValueError(f"No valid points after filtering for '{object_name}'")

        # an unsubsampled multiview payload stalls the GraspNet service
        pc_full = self.subsample_point_cloud(pc_full, max_points=20000)
        pc_segment = self.subsample_point_cloud(pc_segment, max_points=10000)

        # Contact-GraspNet expects camera-frame clouds; world-frame input yields grasps from under the table.
        head_pose_mat = np.asarray(
            obs[self.camera_name]["pose_mat"], dtype=np.float64
        )
        world_to_cam = np.linalg.inv(head_pose_mat)

        def _world_to_cam_points(pts: np.ndarray) -> np.ndarray:
            pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
            return (world_to_cam @ pts_h.T).T[:, :3]

        # GraspNet candidates are re-ranked geometrically (grasp_candidates), with a deterministic top-down fallback.
        from capsim.integrations.robotwin import grasp_candidates as _gc
        table_z_g = self._table_plane_z()
        if table_z_g is None:
            table_z_g = float(np.percentile(np.asarray(pc_segment)[:, 2], 5))
        geom = _gc.object_geometry(np.asarray(pc_segment), table_z_g)
        try:
            grasp_sample_tf, grasp_scores = self.plan_grasp_from_point_clouds(
                _world_to_cam_points(pc_full), _world_to_cam_points(pc_segment)
            )
            grasp_sample_tf = head_pose_mat @ grasp_sample_tf  # (N,4,4) back to world
        except ValueError as exc:
            print(f"sample_grasp_pose: GraspNet gave no candidates for '{object_name}' ({exc}); geometric fallback")
            grasp_sample_tf, grasp_scores = np.zeros((0, 4, 4)), np.zeros((0,))

        def _to_endpose_grasp(idx: int) -> vtf.SE3:
            g = vtf.SE3.from_matrix(grasp_sample_tf[idx])
            g = g @ vtf.SE3.from_rotation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2)
            )
            return g @ vtf.SE3.from_rotation(
                rotation=vtf.SO3(wxyz=self._ZAPPROACH_TO_TCP_WXYZ)
            )

        # Return the first ranked candidate the planner can reach.
        order = np.argsort(grasp_scores)[::-1][: max(self._FEASIBILITY_TOP_K, 20)]
        raw_cands = []
        for idx in order:
            c = _to_endpose_grasp(int(idx))
            raw_cands.append((np.asarray(c.wxyz_xyz[-3:], float), np.asarray(c.wxyz_xyz[:4], float),
                              float(grasp_scores[idx])))
        ordered = _gc.propose(raw_cands, geom, object_name, table_z_g, cloud=np.asarray(pc_segment),
                              robot_dir=self._ROBOT_DIR)
        print(f"sample_grasp_pose: '{object_name}' h={geom['h']:.3f} w={geom['w']:.3f} l={geom['l']:.3f} "
              f"tall={_gc.is_tall(geom, object_name)} candidates: graspnet {len(raw_cands)} -> ordered {len(ordered)}")
        probes = []
        best_grasp = None
        best_tags = None
        # probes cost seconds each: at most 8 ranked candidates, plus always the deterministic fallback
        _det = lambda c: any(t.startswith("fallback") for t in c[3])
        probe_list = [c for c in ordered if not _det(c)][:8] + [c for c in ordered if _det(c)]
        for pos, quat, score, tags in probe_list:
            ok = self._grasp_plan_feasible(pos, quat, hand=hand)
            probes.append({"tags": list(tags), "score": float(score), "plan_ok": bool(ok)})
            if ok:
                best_grasp = (np.asarray(pos, float), np.asarray(quat, float))
                best_tags = tags
                break
        if best_grasp is None:
            raise ValueError(
                f"No reachable grasp candidate for '{object_name}' among the top "
                f"{len(probes)} proposals - call sample_grasp_pose again "
                f"(candidates are stochastic) or try a different object description."
            )

        print(f"sample_grasp_pose in {time.time() - start_time} seconds ({'/'.join(best_tags)})")
        print(f"Grasp sample position for {object_name}: {best_grasp[0]}")
        print(f"Grasp sample quaternion wxyz for {object_name}: {best_grasp[1]}")

        return best_grasp[0], best_grasp[1]

    # Superlatives and spatial ordinals are resolved geometrically (enumerate instances, measure, rank):
    # the pointer picks the wrong instance for "the largest X".
    _SUPERLATIVE_RE = _vocab.words_re(_vocab.section("superlatives"))
    # Ranked by world x (-x = robot's left); bare "left"/"right" are excluded ("left shoe" is a name).
    _SPATIAL_RANK = dict(_vocab.section("spatial_ordinals"))
    _SPATIAL_RE = _vocab.words_re(_SPATIAL_RANK)
    _SPATIAL_CONTINUITY_M = _P["spatial"]["continuity"]
    _SPATIAL_OCCLUDE_M = _P["spatial"]["occlude"]
    _POINT_DEDUPE_PX = _P["fusion"]["point_dedupe_px"]
    _INSTANCE_MERGE_M = _P["fusion"]["instance_merge"]
    _RANK_REPEATS = _P["superlative"]["rank_repeats"]
    _SIZE_KEY = _P["superlative"]["size_key"]
    _SUPERLATIVE_ON = bool(_P["superlative"]["enabled"])
    _SPATIAL_ON = bool(_P["spatial"]["enabled"])

    @staticmethod
    def _min_area_rect_angle(pts_xy) -> float | None:
        r = RoboTwinBaseApi._min_area_rect(pts_xy)
        return None if r is None else r[0]

    @staticmethod
    def _min_area_rect(pts_xy):
        """(angle_rad, width, length) of the min-area enclosing rectangle (numpy/scipy, no cv2), or None."""
        try:
            from scipy.spatial import ConvexHull
            P = np.asarray(pts_xy, dtype=np.float64)
            if len(P) > 3000:                      # deterministic stride
                P = P[:: (len(P) // 3000) + 1]
            H = P[ConvexHull(P).vertices]
            best = None
            for i in range(len(H)):
                e = H[(i + 1) % len(H)] - H[i]
                if np.hypot(e[0], e[1]) < 1e-9:
                    continue
                th = float(np.arctan2(e[1], e[0]))
                c, s_ = np.cos(-th), np.sin(-th)
                Q = H @ np.array([[c, s_], [-s_, c]])
                w = float(Q[:, 0].max() - Q[:, 0].min())
                l = float(Q[:, 1].max() - Q[:, 1].min())
                if best is None or w * l < best[1] * best[2]:
                    best = (th, w, l)
            return best
        except Exception as exc:  # noqa: BLE001
            print(f"_min_area_rect failed: {exc}")
            return None

    def _size_factors(self, p3):
        """(top-face footprint m^2, height above the table plane m or None) of one object cloud."""
        a = np.asarray(p3)
        x_lo, x_hi = np.percentile(a[:, 0], [5, 95])
        y_lo, y_hi = np.percentile(a[:, 1], [5, 95])
        fp = float((x_hi - x_lo) * (y_hi - y_lo))
        z_top = float(np.percentile(a[:, 2], 95))
        top = a[a[:, 2] > z_top - 0.01]
        if len(top) >= 30:
            rect = self._min_area_rect(top[:, :2])
            if rect is not None and rect[1] > 1e-4 and rect[2] > 1e-4:
                fp = float(rect[1] * rect[2])
        table_z = self._table_plane_z()
        h = None if table_z is None else z_top - table_z
        if h is not None and h <= 0.002:
            h = None
        return fp, h

    @staticmethod
    def _combine_size(fp, h, mode):
        if mode == "footprint" or h is None:
            return fp
        return h if mode == "height" else fp * h

    def _size_key(self, p3) -> float:
        """Size ordering key in the episode's size mode (see _choose_size_mode)."""
        fp, h = self._size_factors(p3)
        mode = os.environ.get("CAPX_SIZE_KEY") or self._SIZE_KEY or getattr(self, "_size_mode", "volume")
        return self._combine_size(fp, h, mode)

    def _choose_size_mode(self, samples) -> str:
        """Size key with the best (min between-object log gap) / (within-object noise) over {rank: [(fp, h)]}."""
        best, best_score = "volume", -np.inf
        for mode in ("height", "footprint", "volume"):
            vals = {}
            for i, fac in samples.items():
                v = [self._combine_size(fp, h, mode) for fp, h in fac
                     if not (mode != "footprint" and h is None)]
                v = [x for x in v if x and x > 0]
                if len(v) < 1:
                    vals = {}
                    break
                vals[i] = v
            if len(vals) < 2:
                continue
            # divide logs by the key's length dimension so the keys are comparable
            dim = {"height": 1.0, "footprint": 2.0, "volume": 3.0}[mode]
            med = {i: float(np.median(np.log(v))) / dim for i, v in vals.items()}
            disp = [float(np.median(np.abs(np.log(v) / dim - med[i])))
                    for i, v in vals.items()]
            order = sorted(med.values())
            gap = min(b - a for a, b in zip(order, order[1:]))
            noise = max(float(np.median(disp)), 1e-4)
            score = gap / noise
            print(f"[superlative] size-key candidate {mode:9s} gap={gap:.3f} "
                  f"noise={noise:.3f} score={score:.1f}")
            if score > best_score:
                best, best_score = mode, score
        return best

    def _table_plane_z(self) -> float | None:
        """Table height from the head-camera cloud (modal 5 mm z bin in the workspace), cached; None if too sparse."""
        cached = getattr(self, "_table_z_cache", None)
        if cached is not None:
            return cached
        try:
            obs = self.get_observation()
            cam = obs[self.camera_name]
            pts = depth_to_pointcloud(cam["images"]["depth"], cam["intrinsics"],
                                      subsample_factor=4)
            ones = np.ones((len(pts), 1))
            w = (cam["pose_mat"] @ np.concatenate([pts, ones], axis=1).T).T[:, :3]
            lo, hi = self._workspace_bounds()
            m = ((w[:, 0] > lo[0]) & (w[:, 0] < hi[0])
                 & (w[:, 1] > lo[1]) & (w[:, 1] < hi[1])
                 & (w[:, 2] > lo[2] - 0.05) & (w[:, 2] < hi[2]))
            z = w[m][:, 2]
            if len(z) < 500:
                return None
            nbins = max(4, int((z.max() - z.min()) / 0.005) + 1)
            hist, edges = np.histogram(z, bins=nbins)
            k = int(np.argmax(hist))
            self._table_z_cache = float((edges[k] + edges[k + 1]) / 2)
            return self._table_z_cache
        except Exception as exc:  # noqa: BLE001
            print(f"_table_plane_z failed: {exc}")
            return None

    def _point_masks(self, rgb: np.ndarray, point) -> list[dict[str, Any]]:
        """Point-prompted SAM3 masks."""
        return self.segment_sam3_point_prompt(rgb, point)

    def _enumerate_instances(self, obs, cam_name, noun, want):
        """Distinct ``noun`` instances in one camera view (Molmo points -> SAM3 -> backprojection -> gates)."""
        cam = obs.get(cam_name)
        if not cam or "rgb" not in cam.get("images", {}):
            return []
        rgb = cam["images"]["rgb"]
        depth = cam["images"]["depth"]
        intrinsics = cam["intrinsics"]
        extrinsics = cam["pose_mat"]
        uniq: list[tuple[float, float]] = []
        # Only same-class retry queries: an off-class query makes Molmo point at shadows.
        is_block = bool(re.search(r"\b(block|blocks|cube|cubes)\b", noun, re.IGNORECASE))
        alt = noun[:-1] if noun.endswith("s") else noun + "s"
        queries = [noun, f"colored {noun}"] + (["cube"] if is_block else [alt])
        for q_noun in queries:
            for p in self.molmo_all_points_fn(Image.fromarray(rgb), q_noun):
                if p is None or p[0] is None:
                    continue
                if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                       > self._POINT_DEDUPE_PX ** 2 for q in uniq):
                    uniq.append((float(p[0]), float(p[1])))
            if is_block and len(uniq) >= max(want, 3):
                break
        d2 = np.squeeze(np.asarray(depth))
        pts_cam = depth_to_pointcloud(d2, intrinsics, subsample_factor=1,
                                      filter_invalid=False)
        ones = np.ones((len(pts_cam), 1))
        pts_world = (extrinsics @ np.concatenate([pts_cam, ones], axis=1).T).T[:, :3]
        valid_flat = (d2 > 0).flatten()
        lo, hi = self._workspace_bounds()
        table_z = None if is_block else self._table_plane_z()
        img_px = float(d2.shape[0] * d2.shape[1])
        instances = []
        for pt in uniq:
            masks = self._point_masks(rgb, pt)
            if not masks:
                continue
            md = max(masks, key=lambda x: x["score"])
            if md["score"] < self._SEG_MIN_SCORE:
                print(f"[enum] '{noun}' point {tuple(int(v) for v in pt)} dropped: score {md['score']:.2f}")
                continue
            if not is_block:
                # one object prompted twice (rim vs centre of a bowl): dedupe by mask overlap
                m_new = np.asarray(md["mask"], dtype=bool)
                dup = None
                for inst in instances:
                    m_old = np.asarray(inst["mask"], dtype=bool)
                    inter = float(np.logical_and(m_new, m_old).sum())
                    if inter <= 0:
                        continue
                    union = float(np.logical_or(m_new, m_old).sum())
                    contain = inter / max(min(m_new.sum(), m_old.sum()), 1.0)
                    if inter / union > 0.3 or contain > 0.7:
                        dup = inst
                        break
                if dup is not None:
                    if md["score"] <= dup["score"]:
                        continue
                    instances = [i for i in instances if i is not dup]   # not .remove(): dict == compares arrays
                if m_new.sum() > 0.06 * img_px:
                    print(f"[enum] '{noun}' point {tuple(int(v) for v in pt)} dropped: mask {100*m_new.sum()/img_px:.0f}% of the image")
                    continue                    # SAM3 bleed over the textured table
            mask_flat = md["mask"].flatten() & valid_flat
            n = min(len(pts_world), len(mask_flat))
            p3 = pts_world[:n][mask_flat[:n]]
            p3, _ = self.filter_noise(p3)
            if len(p3) < 30:
                continue
            centroid = p3.mean(axis=0)
            span = p3.max(axis=0) - p3.min(axis=0)
            max_extent = self._MAX_OBJECT_EXTENT if is_block else 0.25
            if (np.any(centroid < lo) or np.any(centroid > hi)
                    or np.any(span > max_extent)):
                print(f"[enum] '{noun}' point {tuple(int(v) for v in pt)} dropped: centroid {np.round(centroid, 2).tolist()} span {np.round(span, 2).tolist()}")
                continue
            if table_z is not None and float(np.percentile(p3[:, 2], 95)) < table_z + 0.015:
                print(f"[enum] '{noun}' point {tuple(int(v) for v in pt)} dropped: flat on the table")
                continue                        # flat on the table = a shadow, not an object
            if not is_block:
                # a cloud within 8 cm of a closed gripper is the held object, not a table instance
                near_tip = False
                sub = p3[:: max(1, len(p3) // 1000)]
                for h_, (pose_, tip_, closed_) in self._all_fingertips().items():
                    if closed_ and float(np.min(np.linalg.norm(sub - tip_, axis=1))) < 0.08:
                        near_tip = True
                if near_tip:
                    print(f"[enum] '{noun}' point {tuple(int(v) for v in pt)} dropped: at a closed gripper (held)")
                    continue
            instances.append({"mask": md["mask"], "score": md["score"],
                              "points_3d": p3, "centroid": centroid,
                              "size": self._size_key(p3)})
        if not is_block and len(instances) >= 3:
            # drop phantom (shadow / table bleed) instances far larger than the median; needs >= 3 instances
            fps = [self._size_factors(i["points_3d"])[0] for i in instances]
            keep = []
            for i, inst in enumerate(instances):
                others = [f for j, f in enumerate(fps) if j != i]
                if fps[i] > 6.0 * float(np.median(others)):
                    print(f"[superlative] dropping giant '{noun}' instance "
                          f"({fps[i] * 1e4:.0f} cm2 vs median {np.median(others) * 1e4:.0f})")
                    continue
                keep.append(inst)
            instances = keep
        return instances

    def _augment_instance_multiview(self, inst, obs):
        """Fuse aux views into one instance, prompting each aux camera with the projected centroid (keeps identity)."""
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return inst
        base_pts = np.asarray(inst["points_3d"])
        if len(base_pts) < 30:
            return inst
        tree = cKDTree(base_pts)
        merged = [base_pts]
        for cam_name in self.aux_camera_names:
            cam = obs.get(cam_name)
            if not cam or "rgb" not in cam.get("images", {}):
                continue
            K = cam.get("intrinsics")
            T = cam.get("pose_mat")
            depth = cam["images"].get("depth")
            if K is None or T is None or depth is None:
                continue
            d2 = np.squeeze(np.asarray(depth))
            c = np.linalg.inv(T) @ np.append(inst["centroid"], 1.0)
            if c[2] <= 0.05:
                continue
            u = K[0][0] * c[0] / c[2] + K[0][2]
            v = K[1][1] * c[1] / c[2] + K[1][2]
            H, W = d2.shape[:2]
            if not (0 <= u < W and 0 <= v < H):
                continue
            masks = self._point_masks(cam["images"]["rgb"], (float(u), float(v)))
            if not masks:
                continue
            md = max(masks, key=lambda x: x["score"])
            if md["score"] < self._SEG_MIN_SCORE:
                continue
            pts_cam = depth_to_pointcloud(d2, K, subsample_factor=1,
                                          filter_invalid=False)
            ones = np.ones((len(pts_cam), 1))
            ptsw = (np.asarray(T) @ np.concatenate([pts_cam, ones], axis=1).T).T[:, :3]
            mf = md["mask"].flatten() & (d2 > 0).flatten()
            n = min(len(ptsw), len(mf))
            p3 = ptsw[:n][mf[:n]]
            if len(p3) == 0:
                continue
            d, _ = tree.query(p3, k=1)
            if d.min() > self._VIEW_CONSISTENCY_M:
                continue
            merged.append(p3[d < self._POINT_MERGE_M])
        if len(merged) > 1:
            fused = np.concatenate(merged)
            fused, _ = self.filter_noise(fused)
            if len(fused) >= 30:
                inst = dict(inst)
                inst["points_3d"] = fused
                inst["centroid"] = fused.mean(axis=0)
                inst["size"] = self._size_key(fused)
        return inst

    def _resolve_size_superlative(self, text_prompt: str, sup_word: str) -> dict[str, Any]:
        """Resolve a size superlative geometrically; raises ValueError when too few instances are measurable."""
        # cached per sim step so one perception round ranks from one measurement
        cache_key = getattr(self._rt_env, "_sim_step_count", None)
        cached = getattr(self, "_superlative_cache", None)
        if cached is not None and cached[0] == cache_key:
            return self._pick_superlative(cached[1], text_prompt, sup_word)
        obs = self.get_observation()
        noun = self._SUPERLATIVE_RE.sub(" ", text_prompt)
        noun = re.sub(r"\bthe\b", " ", noun, flags=re.IGNORECASE)
        noun = re.sub(r"\s+", " ", noun).strip() or "object"

        want = 3 if getattr(self, "_sup_track", None) is None else 1
        instances = self._enumerate_instances(obs, self.camera_name, noun, want)
        merged = []
        for it in sorted(instances, key=lambda x: -x["score"]):
            if all(np.linalg.norm(it["centroid"][:2] - m["centroid"][:2])
                   > self._INSTANCE_MERGE_M for m in merged):
                merged.append(it)
        # head view under-enumerated: back it up with the front camera
        if len(merged) < max(want, 3) and self._front_camera in obs:
            for it in sorted(self._enumerate_instances(
                    obs, self._front_camera, noun, want), key=lambda x: -x["score"]):
                if all(np.linalg.norm(it["centroid"][:2] - m["centroid"][:2])
                       > 0.05 for m in merged):
                    merged.append(it)
        merged = [self._augment_instance_multiview(m, obs) for m in merged]
        merged.sort(key=lambda x: x["size"])

        # Lock rank -> object at the first full enumeration, then track by re-association: once one object is
        # grasped, re-ranking would silently give its name to another.
        track = getattr(self, "_sup_track", None)
        if track is None:
            need_init = 3 if sup_word.lower() == "medium" else 2
            if len(merged) < need_init:
                raise ValueError(
                    f"size-superlative '{text_prompt}': only {len(merged)} "
                    f"distinct instances visible for initial ranking "
                    f"(need {need_init})")
            # a wrong lock is unrecoverable: rank by the median over repeated enumerations
            reps = int(os.environ.get("CAPX_RANK_REPEATS", self._RANK_REPEATS))
            self._sup_identical = False
            if reps > 1:
                fac = {i: [self._size_factors(m["points_3d"])]
                       for i, m in enumerate(merged)}
                for _ in range(reps - 1):
                    extra = self._enumerate_instances(obs, self.camera_name, noun, want)
                    extra = [self._augment_instance_multiview(m, obs) for m in extra]
                    for i, m in enumerate(merged):          # match by centroid
                        near = min(extra, key=lambda e: float(np.linalg.norm(
                            e["centroid"][:2] - m["centroid"][:2])), default=None)
                        if near is not None and np.linalg.norm(
                                near["centroid"][:2] - m["centroid"][:2]) < self._INSTANCE_MERGE_M:
                            fac[i].append(self._size_factors(near["points_3d"]))
                if not (os.environ.get("CAPX_SIZE_KEY") or self._SIZE_KEY):
                    self._size_mode = self._choose_size_mode(fac)
                mode = os.environ.get("CAPX_SIZE_KEY") or self._SIZE_KEY or self._size_mode
                for i, m in enumerate(merged):
                    v = [self._combine_size(fp, h, mode) for fp, h in fac[i]]
                    v = [x for x in v if x and x > 0]
                    if v:
                        m["size"] = float(np.median(v))
                merged.sort(key=lambda x: x["size"])
                # identical objects: size order is noise, so label by position and track by distance
                dim = {"height": 1.0, "footprint": 2.0, "volume": 3.0}[mode]
                logs = [np.log(max(m["size"], 1e-9)) / dim for m in merged]
                gap = min((b - a for a, b in zip(logs, logs[1:])), default=1.0)
                self._sup_identical = bool(gap < 0.05)
                if self._sup_identical:
                    merged.sort(key=lambda x: (round(float(x["centroid"][1]), 2),
                                               float(x["centroid"][0])))
                    print(f"[superlative] sizes within noise (gap {gap:.3f}) -> "
                          f"labelling by position, tracking by distance")
                print(f"[superlative] ranking locked: key={mode} from {reps} samples, "
                      f"sizes {[round(m['size'], 5) for m in merged]} "
                      f"(n per rank {[len(fac[i]) for i in sorted(fac)]})")
            track = {r: {"centroid": m["centroid"].copy(), "size": m["size"],
                         "inst": m}
                     for r, m in enumerate(merged)}
            self._sup_track = track
            assign = dict(enumerate(merged))
        else:
            # size similarity first (invariant to being moved), distance as tiebreak
            identical = bool(getattr(self, "_sup_identical", False))

            def cost(rec, m):
                size_term = abs(np.log(max(m["size"], 1e-6) / max(rec["size"], 1e-6)))
                dist_term = float(np.linalg.norm(m["centroid"][:2] - rec["centroid"][:2]))
                if identical:
                    return dist_term + 0.2 * size_term if dist_term < 0.12 else 9.0
                # an unmoved instance keeps its rank even if a parked gripper contaminates its size
                if dist_term < 0.05:
                    return 0.5 * dist_term
                return size_term + 0.5 * dist_term
            held = set(getattr(self, "_held_ranks", {}).values())
            pairs = sorted(
                (cost(rec, m), r, mi)
                for r, rec in track.items() if r not in held
                for mi, m in enumerate(merged))
            assign, used_r, used_m = {}, set(), set()
            print("[superlative] re-associate: " + "; ".join(
                f"rank{r}<-inst{mi} cost={c:.2f}" for c, r, mi in pairs[:6]))
            for c, r, mi in pairs:
                if r in used_r or mi in used_m or c > 1.0:
                    continue
                assign[r] = merged[mi]
                used_r.add(r)
                used_m.add(mi)
            # nested objects show as one outline: an unassigned rank within 6 cm shares that instance
            for r, rec in track.items():
                if r in assign or r in held:
                    continue
                near = [(float(np.linalg.norm(m["centroid"][:2] - rec["centroid"][:2])), mi)
                        for mi, m in enumerate(merged)]
                near = [(d, mi) for d, mi in near if d < 0.06]
                if near:
                    d, mi = min(near)
                    assign[r] = merged[mi]
                    print(f"[superlative] rank {r} shares instance {mi} (nested, {d*100:.1f} cm)")
            for r, m in assign.items():
                # keep the size from track init: per-call estimates flicker and would flip size-primary matching
                self._sup_track[r] = {"centroid": m["centroid"].copy(),
                                      "size": track[r]["size"], "inst": m}
        self._superlative_cache = (cache_key, assign)
        return self._pick_superlative(assign, text_prompt, sup_word)

    @staticmethod
    def _shadow_ser(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {k: RoboTwinBaseApi._shadow_ser(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [RoboTwinBaseApi._shadow_ser(v) for v in x]
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        return x

    def _shadow_log(self, fn: str, args: dict, result) -> None:
        """Log obs, endposes, gripper state and the oracle result per call to CAPX_SHADOW_DIR for offline diffing."""
        d = os.environ.get("CAPX_SHADOW_DIR")
        if not d:
            return
        try:
            import pickle
            from pathlib import Path as _P
            dp = _P(d)
            dp.mkdir(parents=True, exist_ok=True)
            i = getattr(self, "_shadow_i", 0)
            self._shadow_i = i + 1
            step = getattr(self._rt_env, "_sim_step_count", -1)
            obs_key = f"obs_{step}.pkl"
            if not (dp / obs_key).exists():
                with open(dp / obs_key, "wb") as fh:
                    pickle.dump(self.get_observation(), fh)
            endposes = {}
            for h in ("left", "right"):
                try:
                    endposes[h] = np.asarray(
                        self.task.get_arm_pose(self._arm(h))).tolist()
                except Exception:
                    endposes[h] = None
            grip = {h: bool(st.get("closed"))
                    for h, st in getattr(self, "_grip_state", {}).items()}
            try:  # offline replay needs the randomized table pose (_workspace_bounds)
                table_pos = np.asarray(
                    self.task.table.get_pose().p).tolist()
            except Exception:
                table_pos = None
            rec = {"i": i, "fn": fn, "args": self._shadow_ser(args),
                   "obs": obs_key, "sim_step": int(step),
                   "endposes": endposes, "grip": grip, "table_pos": table_pos,
                   "result": self._shadow_ser(result)}
            with open(dp / "calls.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"shadow log failed: {exc}")

    def _note_gripper(self, hand: str, closed: bool) -> None:
        """Record gripper state and actuation-time endpose, and update held-name / held-rank routing."""
        try:
            pose = np.asarray(self.task.get_arm_pose(self._arm(hand)),
                              dtype=np.float64)
        except Exception:
            pose = None
        if not hasattr(self, "_grip_state"):
            self._grip_state = {}
        self._grip_state[hand] = {"closed": closed, "endpose": pose}
        try:
            self._note_held_names(hand, closed, pose)
        except Exception as exc:  # noqa: BLE001
            print(f"[held] routing failed: {exc}")
        # A HELD rank is served only proprioceptively and excluded from visible-instance matching.
        if not hasattr(self, "_held_ranks"):
            self._held_ranks = {}
        track = getattr(self, "_sup_track", None)
        ap = self._gripper_aperture(hand) if closed else None
        if closed and ap is not None and ap < self._APERTURE_EMPTY and track:
            print(f"[superlative] gripper {hand} closed on air (aperture {ap:.3f}) -> no rank marked HELD")
        elif closed and pose is not None and track:
            free = [r for r in track if r not in self._held_ranks.values()]
            if free:
                # prefer the last resolved rank (queried right before grasping), else nearest to the fingertip
                lr = getattr(self, "_last_resolved_rank", None)
                try:
                    _, tip_xy = self._fingertip_of(hand)
                    tip_xy = tip_xy[:2]
                except Exception:
                    tip_xy = pose[:2]
                if lr in free and float(np.linalg.norm(
                        track[lr]["centroid"][:2] - tip_xy)) > 0.15:
                    print(f"[superlative] gripper {hand} closed far from rank {lr} "
                          f"-> not marking it HELD")
                    lr = None
                if lr in free:
                    self._held_ranks[hand] = lr
                    print(f"[superlative] gripper {hand} closed -> rank {lr} "
                          f"marked HELD (from last resolve)")
                else:
                    r_near = min(free, key=lambda r: float(np.linalg.norm(
                        track[r]["centroid"][:2] - tip_xy)))
                    d = float(np.linalg.norm(
                        track[r_near]["centroid"][:2] - tip_xy))
                    if d < 0.15:
                        self._held_ranks[hand] = r_near
                        print(f"[superlative] gripper {hand} closed -> rank "
                              f"{r_near} marked HELD (nearest, d={d:.3f})")
        elif not closed:
            r_rel = self._held_ranks.pop(hand, None)
            if r_rel is not None:
                self._superlative_cache = None
                # move the track to the release point so the next enumeration re-associates it
                try:
                    if track and r_rel in track and pose is not None:
                        _, tip = self._fingertip_of(hand)
                        rec = track[r_rel]
                        d = tip - rec["centroid"]
                        if "inst" in rec and rec["inst"].get("points_3d") is not None:
                            pts_r = np.asarray(rec["inst"]["points_3d"]) + d
                            tz = self._table_plane_z()
                            if tz is not None:          # released objects settle onto their support
                                zb = float(np.percentile(pts_r[:, 2], 5))
                                if zb > tz + 0.005:
                                    pts_r = pts_r - np.array([0.0, 0.0, zb - tz])
                            rec["inst"]["points_3d"] = pts_r
                            rec["inst"]["centroid"] = pts_r.mean(axis=0)
                            rec["centroid"] = pts_r.mean(axis=0)
                        else:
                            rec["centroid"] = tip.copy()
                        print(f"[superlative] {hand} opened -> rank {r_rel} placed at the fingertip "
                              f"{np.round(tip, 3).tolist()}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[superlative] release update failed: {exc}")
        self._shadow_log("gripper_note",
                         {"hand": hand, "closed": closed},
                         {"endpose": pose})

    def _inhand_estimate(self, idx: int, text_prompt: str):
        """Last-seen cloud of a held rank, translated by the holding hand's endpose motion since close."""
        grip = getattr(self, "_grip_state", {})
        rec = getattr(self, "_sup_track", {}).get(idx)
        if rec is None or "inst" not in rec:
            return None
        holders = [h for h, r in getattr(self, "_held_ranks", {}).items()
                   if r == idx and grip.get(h, {}).get("closed")
                   and grip[h].get("endpose") is not None]
        if len(holders) == 1:
            hand = holders[0]
        else:
            return None
        ap = self._gripper_aperture(hand)
        if ap is not None and ap < self._APERTURE_EMPTY:
            print(f"[superlative] '{text_prompt}' rank {idx}: {hand} jaws are empty (aperture {ap:.3f}) -> not served in-hand")
            return None
        try:
            cur = np.asarray(self.task.get_arm_pose(self._arm(hand)),
                             dtype=np.float64)
        except Exception:
            return None
        delta = cur[:3] - grip[hand]["endpose"][:3]
        inst = rec["inst"]
        pts = np.asarray(inst["points_3d"]) + delta
        centroid = rec["centroid"] + delta
        self._sup_track[idx]["centroid"] = centroid
        print(f"[superlative] '{text_prompt}' -> IN-HAND estimate ({hand}), "
              f"delta {np.round(delta, 3).tolist()}")
        self._last_served_inhand = text_prompt
        return {"head_mask": inst["mask"], "points_3d": pts,
                "head_points_3d": pts, "head_score": inst["score"],
                "camera_masks": {self.camera_name: inst["mask"]},
                "camera_scores": {self.camera_name: inst["score"]},
                "camera_points_3d": {self.camera_name: pts}}

    def _occluded_estimate(self, idx: int, text_prompt: str):
        """Last-seen cloud of a rank occluded by an arm hovering within 15 cm while no gripper is closed."""
        rec = getattr(self, "_sup_track", {}).get(idx)
        if rec is None or "inst" not in rec:
            return None
        grip = getattr(self, "_grip_state", {})
        if any(st.get("closed") for st in grip.values()):
            return None
        near = False
        for hand in ("left", "right"):
            try:
                pose = np.asarray(self.task.get_arm_pose(self._arm(hand)), dtype=np.float64)
            except Exception:
                continue
            probes = [pose[:2]]
            try:
                q = pose[3:7]
                R = SciRotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                probes.append((pose[:3] + self._FINGERTIP_OFFSET_M * R[:, 0])[:2])   # fingertip: gripper bias along the approach axis
            except Exception:
                pass
            if any(float(np.linalg.norm(pr - rec["centroid"][:2])) < 0.15 for pr in probes):
                near = True
        if not near:
            return None
        inst = rec["inst"]
        pts = np.asarray(inst["points_3d"])
        print(f"[superlative] '{text_prompt}' -> rank {idx} occluded by our arm, "
              f"serving last-seen cloud")
        return {"head_mask": inst["mask"], "points_3d": pts,
                "head_points_3d": pts, "head_score": inst["score"],
                "camera_masks": {self.camera_name: inst["mask"]},
                "camera_scores": {self.camera_name: inst["score"]},
                "camera_points_3d": {self.camera_name: pts}}

    def _pick_superlative(self, assign: dict, text_prompt: str, sup_word: str) -> dict[str, Any]:
        w = sup_word.lower()
        n_ranks = len(getattr(self, "_sup_track", {})) or 3
        if w in ("largest", "biggest"):
            idx = n_ranks - 1
        elif w in ("smallest", "tiniest"):
            idx = 0
        else:
            idx = n_ranks // 2
        if idx not in assign:
            est = self._inhand_estimate(idx, text_prompt)
            if est is not None:
                return est
            est = self._occluded_estimate(idx, text_prompt)
            if est is not None:
                return est
            raise ValueError(
                f"size-superlative '{text_prompt}': rank {idx} not currently "
                f"visible (likely held in the gripper) - re-query after release")
        chosen = assign[idx]
        self._last_resolved_rank = idx
        print(f"[superlative] '{text_prompt}' -> visible {sorted(assign)}, "
              f"areas {[round(assign[r]['size'] * 1e4, 1) for r in sorted(assign)]} cm2,"
              f" rank {idx}")
        return {"head_mask": chosen["mask"],
                "points_3d": np.asarray(chosen["points_3d"]),
                "head_points_3d": np.asarray(chosen["points_3d"]),
                "head_score": chosen["score"],
                "camera_masks": {self.camera_name: chosen["mask"]},
                "camera_scores": {self.camera_name: chosen["score"]},
                "camera_points_3d": {self.camera_name: np.asarray(chosen["points_3d"])}}

    def _segment_multiview(self, rgb: np.ndarray, text_prompt: str) -> list[dict[str, Any]]:
        """Molmo point -> SAM3 point-prompt masks, falling back to the SAM3 text prompt."""
        points = self.point_prompt_molmo(rgb, text_prompt)
        point = points[text_prompt]
        masks = []
        if point[0] is not None:
            masks = self.segment_sam3_point_prompt(rgb, point)
        if not masks:
            masks = self.segment_sam3_text_prompt(rgb, text_prompt)
        return masks

    def _segment_object_from_language(
        self, image: Image.Image, object_name: str
    ) -> tuple[np.ndarray, tuple[int, int] | None, list[float] | None]:
        """(mask, mask-centre pixel, scores) from the SAM3 text prompt, falling back to Molmo point + SAM3."""
        results = self.sam3_seg_fn(image, text_prompt=object_name)
        if len(results) == 0:
            dets = self.molmo_point_fn(image, objects=[object_name])
            point = dets.get(object_name)
            if point is None or any(coord is None for coord in point):
                return None, None, None
            point_coords = (float(point[0]), float(point[1]))
            results = self.sam3_point_prompt_fn(image, point_coords=point_coords)

        if len(results) == 0:
            return None, None, None

        scores = [result["score"] for result in results]
        best_result = results[np.argmax(scores)]
        mask_bool = best_result["mask"].astype(bool)

        ys, xs = np.where(mask_bool)
        if len(xs) > 0 and len(ys) > 0:
            point_xy = (int(xs.mean()), int(ys.mean()))
        else:
            point_xy = None

        return mask_bool, point_xy, scores

    def get_oriented_bounding_box_from_3d_points(self, points: np.ndarray) -> dict[str, Any]:
        points = points + np.random.normal(0, 0.0001, points.shape)
        o3d_points = o3d.geometry.PointCloud()
        o3d_points.points = o3d.utility.Vector3dVector(points)
        o3d_points, ind = o3d_points.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        obb = o3d_points.get_oriented_bounding_box()
        return {
            "center": obb.center,
            "extent": obb.extent,
            "R": obb.R,
            "quaternion_wxyz": vtf.SO3.from_matrix(obb.R).wxyz,
        }

    # ------------------------------------------------------------------
    # Perception-tier grounding (base_privileged.py overrides the entry points,
    # so the privileged collection API never reaches this section)
    # ------------------------------------------------------------------
    _SEG_SOFT_SCORE = _P["segmentation"]["soft_score"]
    _MASK_FRAC_MIN = _P["segmentation"]["mask_frac_min"]
    _MASK_FRAC_MAX = _P["segmentation"]["mask_frac_max"]
    _MAX_EXTENT_XY = _P["segmentation"]["max_extent_xy"]
    _MAX_EXTENT_Z = _P["segmentation"]["max_extent_z"]
    _HELD_ATTACH_M = _P["held"]["attach"]
    _NAME_ALIASES = _vocab.section("aliases")     # tried only AFTER the literal name fails on the head view
    _COMPOUND_RE = re.compile(r"([a-z]{3,})(" + "|".join(_vocab.section("compound_heads")) + r")")

    def _name_aliases(self, name: str) -> list[str]:
        key = name.strip().lower()
        out = list(self._NAME_ALIASES.get(key, []))
        m_t = re.match(r"^(?:the\s+)?target[\s_]+([a-z]+)$", key)
        if m_t and m_t.group(1) not in ("object", "item", "thing"):
            out.append(m_t.group(1))            # target box -> box
        m = self._COMPOUND_RE.fullmatch(key)
        if m:
            out.append(f"{m.group(1)} {m.group(2)}")
            out.append(m.group(2))
        return [a for a in dict.fromkeys(out) if a != key]

    def _mask_size_ok(self, mask) -> bool:
        m = np.asarray(mask, dtype=bool)
        frac = float(m.sum()) / float(max(m.size, 1))
        return self._MASK_FRAC_MIN <= frac <= self._MASK_FRAC_MAX

    def _fingertip_of(self, hand: str):
        """(endpose[7], fingertip xyz) of a hand, from the robot's TCP pose (endpose + fingertip offset fallback)."""
        pose = np.asarray(self.task.get_arm_pose(self._arm(hand)), dtype=np.float64)
        try:
            fn = getattr(self.task.robot, f"get_{'left' if str(hand).lower() == 'left' else 'right'}_tcp_pose")
            tcp = np.asarray(fn(), dtype=np.float64)
            return pose, tcp[:3]
        except Exception:
            q = pose[3:7]
            R = SciRotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            return pose, pose[:3] + self._FINGERTIP_OFFSET_M * R[:, 0]

    def _all_fingertips(self):
        """{hand: (endpose[7], fingertip xyz, closed: bool)} for both hands (best effort)."""
        out = {}
        grip = getattr(self, "_grip_state", {})
        for hand in ("left", "right"):
            try:
                pose, tip = self._fingertip_of(hand)
                out[hand] = (pose, tip, bool(grip.get(hand, {}).get("closed")))
            except Exception:
                continue
        return out

    def _top_face_centre(self, pts):
        """xy centre of the top-face min-area rectangle, or None."""
        a = np.asarray(pts)
        z_top = float(np.percentile(a[:, 2], 95))
        top = a[a[:, 2] > z_top - 0.015][:, :2]
        if len(top) < 50:
            return None
        rect = self._min_area_rect(top)
        if rect is None:
            return None
        th = rect[0]
        c, s_ = np.cos(-th), np.sin(-th)
        M = np.array([[c, -s_], [s_, c]])
        r = top @ M.T
        mid = np.array([(r[:, 0].min() + r[:, 0].max()) / 2.0, (r[:, 1].min() + r[:, 1].max()) / 2.0])
        return M.T @ mid

    def _gripper_aperture(self, hand: str) -> float | None:
        """Measured (not commanded) normalised opening, 1 = open, 0 = closed; None when unreadable."""
        try:
            robot = self.task.robot
            if str(hand).lower() == "left":
                entity, joints, scale = robot.left_entity, robot.left_gripper, robot.left_gripper_scale
            else:
                entity, joints, scale = robot.right_entity, robot.right_gripper, robot.right_gripper_scale
            base_joint = joints[0][0]
            qpos = np.asarray(entity.get_qpos()).reshape(-1)
            idx = entity.get_active_joints().index(base_joint)
            raw = (float(qpos[idx]) - scale[0]) / (scale[1] - scale[0])
            # fingers meet before the joint limit: re-normalise so 0 = closed on nothing
            val = (raw - self._APERTURE_AIR_RAW) / (1.0 - self._APERTURE_AIR_RAW)
            return float(np.clip(val, 0.0, 1.0))
        except Exception:
            return None

    def get_gripper_aperture(self, hand: str) -> float:
        """Public, tier-independent proprioceptive readback (see the GripperApi
        surface docstring). Raises RuntimeError when the joint is unreadable."""
        ap = self._gripper_aperture(hand)
        if ap is None:
            raise RuntimeError(f"gripper aperture of the {hand} hand is not readable")
        return float(ap)

    def _remember_seen(self, name: str, mask, points_3d, score) -> None:
        """Record the last confident grounding of a plain name."""
        if not hasattr(self, "_last_seen"):
            self._last_seen = {}
        step = getattr(self._rt_env, "_sim_step_count", -1)
        pts = np.asarray(points_3d)
        self._last_seen[name] = {"mask": mask, "points_3d": pts, "score": score,
                                 "centroid": pts.mean(axis=0) if len(pts) else None,
                                 "sim_step": step}

    def _note_held_names(self, hand: str, closed: bool, pose) -> None:
        """On close, mark the plain name nearest the fingertip as held by ``hand``; on open, release it."""
        if not hasattr(self, "_held_names"):
            self._held_names = {}
        if not closed:
            rec_h = self._held_names.pop(hand, None)
            if rec_h is not None:
                seen = getattr(self, "_last_seen", {}).get(rec_h["name"])
                try:
                    if seen is not None and seen.get("centroid") is not None and pose is not None:
                        pose_now, _ = self._fingertip_of(hand)
                        pose0 = rec_h["endpose"]
                        q0, q1 = pose0[3:7], pose_now[3:7]
                        R0 = SciRotation.from_quat([q0[1], q0[2], q0[3], q0[0]]).as_matrix()
                        R1 = SciRotation.from_quat([q1[1], q1[2], q1[3], q1[0]]).as_matrix()
                        Rd = R1 @ R0.T
                        pts = (np.asarray(seen["points_3d"]) - pose0[:3]) @ Rd.T + pose_now[:3]
                        tz = self._table_plane_z()
                        if tz is not None:              # settles onto the support after release
                            zb = float(np.percentile(pts[:, 2], 5))
                            if zb > tz + 0.005:
                                pts = pts - np.array([0.0, 0.0, zb - tz])
                        seen["points_3d"] = pts
                        seen["centroid"] = pts.mean(axis=0)
                        seen["sim_step"] = getattr(self._rt_env, "_sim_step_count", -1)
                except Exception as exc:  # noqa: BLE001
                    print(f"[held] release update failed: {exc}")
                print(f"[held] {hand} opened -> '{rec_h['name']}' released at the fingertip")
            return
        seen = getattr(self, "_last_seen", {})
        if pose is None or not seen:
            return
        ap = self._gripper_aperture(hand)
        if ap is not None and ap < self._APERTURE_EMPTY:
            print(f"[held] {hand} closed on air (aperture {ap:.3f}) -> nothing marked held")
            return
        try:
            _, tip = self._fingertip_of(hand)
        except Exception:
            return
        best = None
        for name, rec in seen.items():
            if rec.get("centroid") is None:
                continue
            pts_r = np.asarray(rec["points_3d"])
            # cloud distance too: long objects are grasped far from their centroid
            d_pts = float(np.min(np.linalg.norm(pts_r[:: max(1, len(pts_r) // 2000)] - tip, axis=1))) if len(pts_r) else 9.0
            d_c = float(np.linalg.norm(rec["centroid"] - tip))
            d = min(d_pts + 0.02, d_c)
            if (d_pts < 0.04 or d_c < self._HELD_ATTACH_M) and (best is None or d < best[0]):
                best = (d, name)
        if best is not None:
            self._held_names[hand] = {"name": best[1], "endpose": np.asarray(pose, dtype=np.float64)}
            print(f"[held] {hand} closed {best[0]*100:.1f} cm from '{best[1]}' -> marked held")

    def _holder_of(self, name: str):
        for hand, rec in getattr(self, "_held_names", {}).items():
            if rec["name"] == name and getattr(self, "_grip_state", {}).get(hand, {}).get("closed"):
                return hand
        return None

    def _inhand_plain_estimate(self, name: str):
        """Last-seen cloud of a held plain name, moved rigidly with the holding hand."""
        hand = self._holder_of(name)
        rec = getattr(self, "_last_seen", {}).get(name)
        if hand is None or rec is None or rec.get("centroid") is None:
            return None
        ap = self._gripper_aperture(hand)
        if ap is not None and ap < self._APERTURE_EMPTY:
            print(f"[held] '{name}': {hand} jaws are empty (aperture {ap:.3f}) -> not served in-hand")
            return None
        try:
            pose_now, _ = self._fingertip_of(hand)
        except Exception:
            return None
        pose0 = self._held_names[hand]["endpose"]
        q0, q1 = pose0[3:7], pose_now[3:7]
        R0 = SciRotation.from_quat([q0[1], q0[2], q0[3], q0[0]]).as_matrix()
        R1 = SciRotation.from_quat([q1[1], q1[2], q1[3], q1[0]]).as_matrix()
        Rd = R1 @ R0.T
        pts = (np.asarray(rec["points_3d"]) - pose0[:3]) @ Rd.T + pose_now[:3]
        print(f"[held] '{name}' served by the {hand} proprioceptive estimate "
              f"(delta {np.round(pose_now[:3] - pose0[:3], 3).tolist()})")
        self._last_served_inhand = name
        return {"head_mask": rec["mask"], "points_3d": pts, "head_points_3d": pts,
                "head_score": rec["score"],
                "camera_masks": {self.camera_name: rec["mask"]},
                "camera_scores": {self.camera_name: rec["score"]},
                "camera_points_3d": {self.camera_name: pts}}

    def _backproject_mask(self, cam, mask):
        depth = cam["images"]["depth"]
        pts_camera = depth_to_pointcloud(depth, cam["intrinsics"], subsample_factor=1, filter_invalid=False)
        ones = np.ones((len(pts_camera), 1))
        pts_world = (np.asarray(cam["pose_mat"]) @ np.concatenate([pts_camera, ones], axis=1).T).T[:, :3]
        mask_flat = np.asarray(mask).flatten() & (np.asarray(depth) > 0).flatten()
        n = min(len(pts_world), len(mask_flat))
        return pts_world[:n][mask_flat[:n]]

    def _plausible_cloud(self, pts, name: str = "") -> tuple[bool, str]:
        """(ok, reason) sanity gate on the cleaned cloud (workspace, extent, table, self-grounding)."""
        if len(pts) == 0:
            return False, "no points"
        clean, _ = self.filter_noise(np.asarray(pts))
        if len(clean) < 10:
            clean = np.asarray(pts)              # the percentile span below already trims flyers
        if len(clean) < 10:
            return False, "too few points"
        lo_p, hi_p = np.percentile(clean, [2, 98], axis=0)
        span = hi_p - lo_p
        centroid = clean.mean(axis=0)
        lo, hi = self._workspace_bounds()
        # widened: floor bins stand beside the table
        lo = lo.copy(); hi = hi.copy()
        lo[:2] -= 0.20; hi[:2] += 0.20; lo[2] -= 0.30; hi[2] += 0.25   # carried objects rise ~0.5 m
        if np.any(centroid < lo) or np.any(centroid > hi):
            return False, (f"centroid {np.round(centroid, 3).tolist()} outside the workspace "
                           f"[{np.round(lo, 2).tolist()} .. {np.round(hi, 2).tolist()}]")
        if span[0] > self._MAX_EXTENT_XY or span[1] > self._MAX_EXTENT_XY or span[2] > self._MAX_EXTENT_Z:
            return False, f"span {np.round(span, 3).tolist()} too large"
        tz = self._table_plane_z()
        floor_name = self._is_floor_name(name)
        if tz is not None and centroid[2] < tz - 0.05 and not floor_name:
            return False, f"centroid z {centroid[2]:.3f} below the table top {tz:.3f}"
        # raw cloud: DBSCAN drops most of a far, obliquely viewed floor object as noise
        floor_bottom = float(np.percentile(np.asarray(pts)[:, 2], 5))
        if tz is not None and floor_name and floor_bottom >= tz - 0.05:
            return False, (f"'{name}' is a floor object but its cloud bottom "
                           f"{floor_bottom:.3f} is not below the table top {tz:.3f}")
        holder = self._holder_of(name) if name else None
        prev = getattr(self, "_last_seen", {}).get(name) if name else None
        tracked = (prev is not None and prev.get("centroid") is not None
                   and float(np.linalg.norm(prev["centroid"][:2] - centroid[:2])) < 0.05)
        if not tracked and holder is None and not floor_name:
            # first grounding only: reject our own open gripper / wrist grounded as the object
            sub = clean[:: max(1, len(clean) // 2000)]
            z_bot = float(np.percentile(clean[:, 2], 5))
            elevated = tz is not None and z_bot > tz + 0.10
            near_closed = False
            for h_, (pose_, tip_, closed_) in self._all_fingertips().items():
                d_tip = float(np.min(np.linalg.norm(sub - tip_, axis=1)))
                d_wrist = float(np.min(np.linalg.norm(sub - pose_[:3], axis=1)))
                if closed_ and d_tip < 0.08:
                    near_closed = True
                if elevated and not closed_ and (d_tip < 0.05 or d_wrist < 0.05) and (
                        float(np.mean(np.linalg.norm(sub - tip_, axis=1) < 0.10)) > 0.7):
                    return False, f"cloud lies on the {h_} gripper (self-grounding)"
        return True, ""

    def _instances_of(self, obs, noun: str, want: int):
        """Head-view instances of ``noun``, deduplicated by xy centroid and cached per sim step."""
        step = getattr(self._rt_env, "_sim_step_count", -1)
        cache = getattr(self, "_indexed_cache", None)
        if cache is None or cache[0] != step or cache[1] != noun:
            inst = self._enumerate_instances(obs, self.camera_name, noun, want)
            merged = []
            for it in sorted(inst, key=lambda x: -x["score"]):
                if all(np.linalg.norm(it["centroid"][:2] - m["centroid"][:2]) > self._INSTANCE_MERGE_M for m in merged):
                    merged.append(it)
            self._indexed_cache = (step, noun, merged)
            cache = self._indexed_cache
        return cache[2]

    @staticmethod
    def _as_segment(it):
        return {"mask": it["mask"], "score": it["score"], "points_3d": it["points_3d"]}

    def _ground_spatial(self, obs, noun: str, word: str):
        """'the leftmost/rightmost/middle <noun>' by world x; None when too few instances are visible."""
        rank = self._SPATIAL_RANK[word.lower()]
        need = 3 if rank == "mid" else 2
        merged = self._instances_of(obs, noun, need)
        # rank only instances still in the workspace, so the phrase stays stable as objects are put away
        try:
            lo, hi = self._workspace_bounds()
            inside = [m for m in merged
                      if np.all(np.asarray(m["centroid"], float) >= lo)
                      and np.all(np.asarray(m["centroid"], float) <= hi)]
            if len(inside) != len(merged):
                print(f"[spatial] '{word} {noun}': {len(merged) - len(inside)} instance(s) outside the workspace")
            merged = inside or merged
        except Exception as exc:                      # noqa: BLE001 - bounds are a refinement
            print(f"[spatial] workspace filter unavailable ({exc})")
        merged = sorted(merged, key=lambda x: float(x["centroid"][0]))
        if len(merged) < need:
            print(f"[spatial] '{word} {noun}': only {len(merged)} instances enumerated, need {need}")
            return None
        it = merged[len(merged) // 2] if rank == "mid" else merged[rank]
        print(f"[spatial] '{word} {noun}' -> instance of {len(merged)} at {np.round(it['centroid'], 3).tolist()} "
              f"(x order {[round(float(m['centroid'][0]), 3) for m in merged]})")
        return self._as_segment(it)

    def get_object_3d_points_and_masks_from_language(
        self,
        text_prompt: str,
        use_multiview: bool = True,
    ) -> dict[str, Any]:
        sup = self._SUPERLATIVE_RE.search(text_prompt)
        if sup and os.environ.get("CAPX_SUPERLATIVE_RESOLVER", "1" if self._SUPERLATIVE_ON else "0") != "0":
            return self._resolve_size_superlative(text_prompt, sup.group(1))
        obs = self.get_observation()

        cameras = [self.camera_name]
        if use_multiview:
            cameras += [c for c in self.aux_camera_names if c in obs]

        camera_data = {}
        head_err = None
        if getattr(self, "_last_served_inhand", None) == text_prompt:
            self._last_served_inhand = None
        held_est = self._inhand_plain_estimate(text_prompt) if self._holder_of(text_prompt) else None
        spatial_anchor = None      # set when a spatial ordinal falls back to plain segmentation by continuity
        spatial_prev = None        # that phrase's last-seen record (served from memory when an arm occludes it)
        step_now = getattr(self._rt_env, "_sim_step_count", -1)
        m_sp = self._SPATIAL_RE.search(text_prompt)
        if m_sp and os.environ.get("CAPX_SPATIAL_RESOLVER", "1" if self._SPATIAL_ON else "0") != "0":
            noun = re.sub(r"\s+", " ", self._SPATIAL_RE.sub(" ", text_prompt)).strip()
            noun = re.sub(r"^the\s+", "", noun, flags=re.IGNORECASE) or "object"
            got = self._ground_spatial(obs, noun, m_sp.group(1))
            if got is not None:
                camera_data[self.camera_name] = got
            elif held_est is None:
                # Unresolvable ordinal: plain segmentation is identity-blind, so accept it only (a) by
                # elimination (one visible, the rest held) or (b) near this phrase's last position.
                need_sp = 3 if self._SPATIAL_RANK[m_sp.group(1).lower()] == "mid" else 2
                vis = self._instances_of(obs, noun, need_sp)
                held_other = sum(1 for rec in getattr(self, "_held_names", {}).values()
                                 if rec.get("name") != text_prompt and noun in str(rec.get("name", "")))
                if len(vis) == 1 and held_other >= need_sp - 1:
                    print(f"[spatial] '{text_prompt}': the only visible '{noun}' with {held_other} other(s) held -> it")
                    camera_data[self.camera_name] = self._as_segment(vis[0])
                    got = camera_data[self.camera_name]
            if got is None and held_est is None:
                prev_sp = getattr(self, "_last_seen", {}).get(text_prompt)
                if prev_sp is None or prev_sp.get("centroid") is None:
                    raise ValueError(
                        f"'{text_prompt}': fewer than the required instances visible to resolve "
                        f"'{m_sp.group(1)}' and no earlier grounding to continue from - not grounded")
                spatial_anchor = np.asarray(prev_sp["centroid"][:2], float)
                spatial_prev = prev_sp
                print(f"[spatial] '{text_prompt}': unresolvable now -> plain '{noun}' accepted only within "
                      f"{self._SPATIAL_CONTINUITY_M * 100:.0f} cm of its last position {np.round(spatial_anchor, 3).tolist()}")

        for cam_name in cameras:
            if cam_name in camera_data:
                continue
            rgb = obs[cam_name]["images"]["rgb"]
            depth = obs[cam_name]["images"]["depth"]
            intrinsics = obs[cam_name]["intrinsics"]
            extrinsics = obs[cam_name]["pose_mat"]

            masks = self._segment_multiview(rgb, text_prompt)
            if held_est is None and cam_name == self.camera_name and (not masks or max(m["score"] for m in masks) < self._SEG_SOFT_SCORE):
                for alias in self._name_aliases(text_prompt):
                    alt = self._segment_multiview(rgb, alias)
                    if alt and max(m["score"] for m in alt) >= self._SEG_MIN_SCORE:
                        print(f"[alias] '{text_prompt}' -> '{alias}' grounded on {cam_name}")
                        masks = alt
                        break
            if not masks:
                # only the head view is mandatory; aux cameras may not have the object in frame
                if cam_name == self.camera_name:
                    head_err = (f"SAM3 segmentation failed for '{text_prompt}' on {cam_name}. "
                                f"No masks returned from either point or text prompt.")
                else:
                    print(f"Skipping {cam_name}: no segmentation for '{text_prompt}'")
                continue
            mask_data = max(masks, key=lambda x: x["score"])
            if cam_name == self.camera_name:
                others = [(n, r["mask"]) for n, r in getattr(self, "_last_seen", {}).items()
                          if n != text_prompt and r.get("sim_step") == step_now and r.get("mask") is not None]
                if others:
                    def _overlaps(mk):
                        a = np.asarray(mk, dtype=bool)
                        for n, om in others:
                            b = np.asarray(om, dtype=bool)
                            if a.shape != b.shape:
                                continue
                            inter = float(np.logical_and(a, b).sum())
                            union = float(np.logical_or(a, b).sum())
                            if inter > 0 and inter / max(union, 1.0) > 0.5:
                                return n
                        return None
                    chosen = None
                    for cand in sorted(masks, key=lambda x: -x["score"]):
                        hit = _overlaps(cand["mask"])
                        if hit is None:
                            chosen = cand
                            break
                        print(f"[exclude] '{text_prompt}' candidate overlaps '{hit}' (grounded this step)")
                    if chosen is None:
                        for alias in self._name_aliases(text_prompt):
                            alt = self._segment_multiview(rgb, alias)
                            for cand in sorted(alt or [], key=lambda x: -x["score"]):
                                if cand["score"] >= self._SEG_MIN_SCORE and _overlaps(cand["mask"]) is None:
                                    chosen = cand
                                    print(f"[exclude] '{text_prompt}' -> alias '{alias}' (score {cand['score']:.2f})")
                                    break
                            if chosen is not None:
                                break
                    if chosen is None:
                        head_err = f"'{text_prompt}' only grounds onto regions already taken by other names this step"
                        continue
                    if chosen is not mask_data:
                        print(f"[exclude] '{text_prompt}' -> next candidate (score {chosen['score']:.2f})")
                    mask_data = chosen

            mask = mask_data["mask"]
            score = mask_data["score"]

            # _SEG_MIN_SCORE gate; the head view may accept [soft, min) with a mask-size gate
            soft_ok = (held_est is None and cam_name == self.camera_name
                       and self._SEG_SOFT_SCORE <= score < self._SEG_MIN_SCORE and self._mask_size_ok(mask))
            if score < self._SEG_MIN_SCORE and not soft_ok:
                if cam_name == self.camera_name:
                    head_err = (f"'{text_prompt}' not confidently segmented on {cam_name} "
                                f"(best SAM3 score {score:.3f} < {self._SEG_MIN_SCORE}); "
                                f"likely not present - try a different object description.")
                else:
                    print(
                        f"Skipping {cam_name}: low-confidence segmentation "
                        f"({score:.3f}) for '{text_prompt}'"
                    )
                continue
            if soft_ok:
                print(f"[soft] '{text_prompt}' accepted on {cam_name} at score {score:.2f} (mask-size gate)")
            if cam_name != self.camera_name and not self._mask_size_ok(mask):
                print(f"Skipping {cam_name}: mask size out of range for '{text_prompt}'")
                continue

            pts_camera = depth_to_pointcloud(
                depth, intrinsics, subsample_factor=1, filter_invalid=False
            )

            pts_homogeneous = np.concatenate([pts_camera, np.ones((len(pts_camera), 1))], axis=1)
            pts_world = (extrinsics @ pts_homogeneous.T).T[:, :3]

            mask_flat = mask.flatten() & (depth > 0).flatten()
            if len(pts_world) != len(mask_flat):
                print(f"Warning: Point cloud size ({len(pts_world)}) doesn't match mask size ({len(mask_flat)}) for {cam_name}")
                min_len = min(len(pts_world), len(mask_flat))
                pts_3d = pts_world[:min_len][mask_flat[:min_len]]
            else:
                pts_3d = pts_world[mask_flat]

            camera_data[cam_name] = {
                "mask": mask,
                "score": score,
                "points_3d": pts_3d
            }

        if self.camera_name not in camera_data:
            # Head view blind (often our own arm): try the holder's wrist, closed hands' wrists, then aux views.
            holder = self._holder_of(text_prompt)
            order = []
            if holder:
                order.append(self._wrist_camera.get(holder))
            for hand, st in getattr(self, "_grip_state", {}).items():
                if st.get("closed"):
                    order.append(self._wrist_camera.get(hand))
            order += list(self.aux_camera_names)
            adopted = None
            for cam_name in dict.fromkeys(order):
                aux = camera_data.get(cam_name)
                if aux is None or len(aux["points_3d"]) == 0:
                    continue
                ok, why = self._plausible_cloud(aux["points_3d"], text_prompt)
                if ok:
                    adopted = cam_name
                    break
                print(f"[aux] {cam_name} grounding of '{text_prompt}' rejected: {why}")
            if adopted is not None:
                print(f"[aux] head view blind for '{text_prompt}' -> using {adopted} "
                      f"(score {camera_data[adopted]['score']:.2f})")
                camera_data[self.camera_name] = camera_data[adopted]
            else:
                est = self._inhand_plain_estimate(text_prompt)
                if est is not None:
                    return est
                raise ValueError(head_err or f"SAM3 segmentation failed for '{text_prompt}'")

        head_data = camera_data[self.camera_name]
        head_pts_3d = head_data["points_3d"]

        points_3d = head_pts_3d

        if use_multiview:
            # merge only aux points near the head anchor cloud: trims mask-bleed tails a union would keep
            view_threshold = self._VIEW_CONSISTENCY_M
            point_threshold = self._POINT_MERGE_M
            merged_any = False
            best_alt: tuple[float, np.ndarray] | None = None
            anchor_tree = (
                cKDTree(self.subsample_point_cloud(head_pts_3d, max_points=20000))
                if len(head_pts_3d) > 0
                else None
            )
            for cam_name in cameras[1:]:
                aux = camera_data.get(cam_name)
                if aux is None or len(aux["points_3d"]) == 0:
                    continue
                aux_pts = aux["points_3d"]
                if anchor_tree is None:
                    if best_alt is None or aux["score"] > best_alt[0]:
                        best_alt = (aux["score"], aux_pts)
                    continue
                dists, _ = anchor_tree.query(aux_pts, k=1)
                if np.min(dists) < view_threshold:
                    points_3d = np.concatenate(
                        [points_3d, aux_pts[dists < point_threshold]]
                    )
                    merged_any = True
                elif best_alt is None or aux["score"] > best_alt[0]:
                    best_alt = (aux["score"], aux_pts)

            if anchor_tree is None and best_alt is not None:
                points_3d = best_alt[1]
                merged_any = True
            # no aux view agreeing with a head grounding keeps the head cloud: views may ground different objects

        if len(points_3d) == 0:
            print(f"Warning: No points found for {text_prompt}")
            points_3d = np.array([]).reshape(0, 3)


        if len(points_3d) > 0 and held_est is not None:
            # while held, a fresh grounding must agree with the proprioceptive estimate
            c_new = np.asarray(points_3d).mean(axis=0)
            c_est = np.asarray(held_est["points_3d"]).mean(axis=0)
            if float(np.linalg.norm(c_new - c_est)) > 0.05:
                print(f"[held] grounding of '{text_prompt}' {np.linalg.norm(c_new - c_est)*100:.1f} cm from "
                      f"the proprioceptive estimate -> serving the estimate")
                return held_est
        if len(points_3d) > 0 and held_est is None:
            # A partial mask under the arm shifts the centroid (read as motion): fuse with the previous cloud.
            prev = getattr(self, "_last_seen", {}).get(text_prompt)
            if prev is not None and prev.get("centroid") is not None and prev.get("sim_step") != step_now:
                c_new = np.asarray(points_3d).mean(axis=0)
                d_prev = float(np.linalg.norm(c_new[:2] - prev["centroid"][:2]))
                arm_over = any(float(np.linalg.norm(tip_[:2] - c_new[:2])) < 0.10
                               for _, (pose_, tip_, closed_) in self._all_fingertips().items())
                shrunk = len(points_3d) < 0.6 * len(prev["points_3d"])
                if d_prev < 0.04 and (arm_over or shrunk):
                    prev_pts = np.asarray(prev["points_3d"])
                    points_3d = np.concatenate([np.asarray(points_3d), prev_pts[:: max(1, len(prev_pts) // max(len(points_3d), 1))]])
                    print(f"[fuse] '{text_prompt}' re-measured under the arm ({d_prev*100:.1f} cm shift) -> fused with the previous cloud")
        if len(points_3d) > 0 and spatial_anchor is not None:
            c_xy = np.median(np.asarray(points_3d)[:, :2], axis=0)
            d_sp = float(np.linalg.norm(c_xy - spatial_anchor))
            if d_sp > self._SPATIAL_CONTINUITY_M:
                # arm over the last position: occluded, serve from memory; otherwise another instance
                arm_over = any(float(np.linalg.norm(np.asarray(tip_)[:2] - spatial_anchor)) < self._SPATIAL_OCCLUDE_M
                               for _, (pose_, tip_, closed_) in self._all_fingertips().items())
                if arm_over and spatial_prev is not None and spatial_prev.get("points_3d") is not None:
                    pts_mem = np.asarray(spatial_prev["points_3d"])
                    print(f"[spatial] '{text_prompt}': plain mask landed {d_sp * 100:.1f} cm away while an arm is over "
                          f"its last position -> served from memory (occluded, assumed static)")
                    return {"head_mask": spatial_prev["mask"], "points_3d": pts_mem, "head_points_3d": pts_mem,
                            "head_score": spatial_prev["score"],
                            "camera_masks": {self.camera_name: spatial_prev["mask"]},
                            "camera_scores": {self.camera_name: spatial_prev["score"]},
                            "camera_points_3d": {self.camera_name: pts_mem}}
                raise ValueError(
                    f"'{text_prompt}': plain segmentation landed {d_sp * 100:.1f} cm from this phrase's last "
                    f"position - a different instance, and too few instances visible to re-rank; not grounded")
            print(f"[spatial] '{text_prompt}' continued by position ({d_sp * 100:.1f} cm from its last grounding)")
        if len(points_3d) > 0:
            ok, why = self._plausible_cloud(points_3d, text_prompt)
            if not ok:
                est = self._inhand_plain_estimate(text_prompt)
                if est is not None:
                    print(f"[held] implausible grounding of '{text_prompt}' ({why}) -> proprioceptive estimate")
                    return est
                raise ValueError(
                    f"'{text_prompt}' grounded to an implausible region ({why}) - likely a false "
                    f"detection; try a different object description."
                )
            self._remember_seen(text_prompt, head_data["mask"], points_3d, head_data["score"])

        return {
            "head_mask": head_data["mask"],
            "points_3d": np.asarray(points_3d),
            "head_points_3d": np.asarray(head_pts_3d),
            "head_score": head_data["score"],
            "camera_masks": {c: d["mask"] for c, d in camera_data.items()},
            "camera_scores": {c: d["score"] for c, d in camera_data.items()},
            "camera_points_3d": {
                c: np.asarray(d["points_3d"]) for c, d in camera_data.items()
            },
        }

    def goto_home_joint_position(self, hand: str) -> bool:
        return self._exec(self.task.back_to_origin(self._arm(hand)))

    def subsample_point_cloud(self, pc: np.ndarray, max_points: int = 10000) -> np.ndarray:
        """Randomly subsample an (N, 3) cloud to at most ``max_points`` points."""
        if len(pc) > max_points:
            return pc[np.random.choice(len(pc), max_points, replace=False)]
        return pc

    def filter_noise(self, points: np.ndarray, colors: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """DBSCAN outlier removal; returns the kept (M, 3) points and matching colors (or None)."""
        points = np.asarray(points)
        eps = 0.005  # m
        min_samples = 10
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(points)
        keep = labels != -1
        # Far, oblique objects are sampled sparser than 5 mm: if most points were dropped, retry with an eps
        # from the cloud's own spacing and keep the largest cluster.
        if keep.sum() < 0.5 * len(points) and len(points) >= 4 * min_samples:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=2).fit(points)
            spacing = float(np.median(nn.kneighbors(points)[0][:, 1]))
            adaptive = max(eps, 3.0 * spacing)
            if adaptive > eps:
                labels2 = DBSCAN(eps=adaptive, min_samples=min_samples).fit_predict(points)
                valid = labels2[labels2 != -1]
                if len(valid):
                    biggest = np.bincount(valid).argmax()
                    keep2 = labels2 == biggest
                    if keep2.sum() > keep.sum():
                        keep = keep2
        filtered_pointcloud = points[keep]
        filtered_colors = np.asarray(colors)[keep] if colors is not None else None
        return filtered_pointcloud, filtered_colors

    def plan_grasp_from_point_clouds(
        self,
        pc_full: np.ndarray,
        pc_segment: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Contact-GraspNet grasps for ``pc_segment`` within the scene ``pc_full`` ((N, 3) each).

        Returns:
            (grasp_sample_tf (N, 4, 4) fingertip poses in the point-cloud frame, grasp_scores (N,)).
        """
        grasp_sample, grasp_scores, _ = self.grasp_net_plan_point_clouds_fn(pc_full, pc_segment, segmap_id=1)

        if len(grasp_sample) == 0:
            raise ValueError(
                "No grasp candidates found - the segmentation may have picked a "
                "false or ungraspable region; try a different object description."
            )

        grasp_sample_tf = (
            vtf.SE3.from_matrix(grasp_sample) @ vtf.SE3.from_translation(np.array([0.0, 0.0, self._FINGERTIP_OFFSET_M]))
        ).as_matrix()
        return grasp_sample_tf, grasp_scores


__all__ = ["RoboTwinBaseApi"]
