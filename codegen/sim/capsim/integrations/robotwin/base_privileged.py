"""Privileged (simulator ground-truth) RoboTwin backends.

* :class:`RoboTwinBasePrivilegedApi` — :class:`RoboTwinBaseApi` with the perception primitives replaced
  by simulator ground truth, so completion measures the program rather than the vision stack; motion and
  geometry stay inherited. Also the oracle the ``scripts/base_api_tests`` probes validate against.
* :class:`RoboTwinGripperPrivilegedApi` — privileged base + unprivileged gripper (jaw motions are never
  privileged).

Prompt invariant: overrides carry no docstrings, so ``ArmApi.__init_subclass__`` copies the surface ones
and the privileged prompt is byte-identical to the perception prompt. Privilege lowers the failure RATE,
never the CONTRACT: ``get_object_pose`` still returns ``(None, None)`` and ``sample_grasp_pose`` still raises.

``privilege`` dials the ground truth: "perception" grounds objects from GT (GraspNet still runs on the GT
segment); "grasp" additionally returns the official annotated grasp, still reachability-checked.

Grounding consults ``object_names.yaml`` (per-task name -> actor attribute) first, so a program can speak
perception language ("the red block"); otherwise actors are scored by query tokens in their names, with
size superlatives ranked by ``config["scale"]`` volume and spatial ordinals by live world x.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import viser.transforms as vtf
from scipy.spatial.transform import Rotation as SciRotation

from capsim.integrations.base_api import ApiBase
from capsim.integrations.robotwin import vocab as _vocab
from capsim.integrations.robotwin.base import RoboTwinBaseApi
from capsim.integrations.robotwin.gripper import RoboTwinGripperApi
from capsim.utils.depth_utils import depth_to_pointcloud

# Full privilege, used by the unit-test oracle; other registrations pass an explicit set.
_DEFAULT_PRIVILEGE = frozenset({"perception", "grasp"})


class RoboTwinBasePrivilegedApi(RoboTwinBaseApi):
    """RoboTwinBaseApi with GT-backed perception/grasp (see module docstring)."""

    # Superlative -> rank among matches sorted by annotated volume (prod config["scale"], ascending).
    # Size superlative / spatial ordinal -> rank (object_vocab.yaml). Ordinals rank by live world x (-x = robot's
    # left); bare "left"/"right" ground via actor names. Ordinals are resolved first, so "middle" is spatial.
    _SUPERLATIVE_RANK: dict[str, Any] = dict(_vocab.section("superlatives"))
    _SPATIAL_RANK: dict[str, Any] = dict(_vocab.section("spatial_ordinals"))

    # Never a referent; demo_randomized cluttered_* names can collide with query nouns.
    _SCENE_EXCLUDE = ("wall", "table")
    _SCENE_EXCLUDE_PREFIXES = ("cluttered",)

    _ORACLE_N_POINTS = 2048     # points per GT object cloud
    _ORACLE_MIN_RADIUS = 0.02   # m; floor for mesh actors whose contact points all sit at the center

    def __init__(
        self,
        env: Any,
        privilege: frozenset[str] | set[str] | tuple[str, ...] = _DEFAULT_PRIVILEGE,
    ) -> None:
        # Skip RoboTwinBaseApi.__init__ (boots all vision); boot only the unprivileged servers.
        ApiBase.__init__(self, env)
        self._rt_env = env
        self._bind_profile(env)
        self._privilege = frozenset(privilege)

        if "perception" not in self._privilege:
            from capsim.integrations.vision.molmo import init_molmo, init_molmo_all_points
            from capsim.integrations.vision.sam3 import init_sam3, init_sam3_point_prompt

            self.sam3_seg_fn = init_sam3()
            self.sam3_point_prompt_fn = init_sam3_point_prompt()
            self.molmo_point_fn = init_molmo()
            self.molmo_all_points_fn = init_molmo_all_points()

        if "grasp" not in self._privilege:
            from capsim.integrations.vision.graspnet import init_contact_graspnet_point_clouds

            self.grasp_net_plan_point_clouds_fn = init_contact_graspnet_point_clouds()

    # ------------------------- GT grounding helpers ------------------------
    def _scene_actors(self) -> dict[str, Any]:
        """Task actors keyed by attribute name; list members become ``<attr><i>`` (i from 1)."""
        from envs.utils.actor_utils import Actor  # type: ignore[import-not-found]

        actors: dict[str, Any] = {}
        for attr, obj in vars(self.task).items():
            if attr in self._SCENE_EXCLUDE or attr.startswith(self._SCENE_EXCLUDE_PREFIXES):
                continue
            if isinstance(obj, Actor):
                actors[attr] = obj
            elif isinstance(obj, (list, tuple)) and obj and all(isinstance(x, Actor) for x in obj):
                for i, x in enumerate(obj, start=1):
                    actors[f"{attr}{i}"] = x
        return actors

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _name_table() -> dict[str, dict[str, str]]:
        """{task: {name: actor attribute}} from object_names.yaml; empty when unreadable."""
        path = Path(__file__).with_name("object_names.yaml")
        try:
            import yaml
            doc = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:                       # noqa: BLE001 - never fail a rollout on it
            print(f"_name_table: {path.name} unavailable ({exc}); using the generic resolver only")
            return {}
        return {str(task): {str(name).strip().lower(): str(actor or "")
                            for name, actor in (names or {}).items()}
                for task, names in doc.items()}

    def _mapped_actor(self, object_name: str, actors: dict[str, Any]) -> tuple[str, Any] | None:
        """The table's actor for this query; None (fall through) for empty or ":"-resolver values."""
        task = type(self.task).__name__
        attr = self._name_table().get(task, {}).get(object_name.strip().lower(), "")
        if attr and ":" not in attr and attr in actors:
            return attr, actors[attr]
        return None

    def _resolve_actor(self, object_name: str) -> tuple[str, Any]:
        """Resolve a description to its GT actor (raises ValueError); CAPX_NAME_TABLE_AUDIT=1 prints table vs generic."""
        actors = self._scene_actors()
        mapped = self._mapped_actor(object_name, actors)
        if mapped is not None:
            if os.environ.get("CAPX_NAME_TABLE_AUDIT"):
                try:
                    generic = self._resolve_actor_generic(object_name, actors)
                except ValueError:
                    generic = None
                agree = generic is not None and generic[0] == mapped[0]
                print(f"[nametable] {'OK  ' if agree else 'DIFF'} {type(self.task).__name__} "
                      f"{object_name!r} table={mapped[0]} "
                      f"generic={'-' if generic is None else generic[0]}")
            return mapped
        return self._resolve_actor_generic(object_name, actors)

    def _resolve_actor_generic(self, object_name: str, actors: dict[str, Any]) -> tuple[str, Any]:
        """The table-free resolver: exact actor name, then token scoring, then a size
        superlative or a spatial ordinal among the actors that matched."""
        # Exact names first: token scoring strips digits and would collapse bottles1/bottles2.
        exact = object_name.strip().lower().replace(" ", "")
        if exact in actors:
            return exact, actors[exact]
        nouns = re.findall(r"[a-z]+", object_name.lower())
        tokens = nouns

        # Most tokens wins: "right shoe" -> right_shoe (2), not left_shoe (1).
        scores = {
            attr: sum(n in f"{attr} {act.get_name()}".lower() for n in nouns)
            for attr, act in actors.items()
        }
        best = max(scores.values(), default=0)
        matched = {a: actors[a] for a, s in scores.items() if s == best and s > 0}
        if not matched:
            raise ValueError(
                f"no scene actor matches {object_name!r} "
                f"(actors: {sorted(actors)})"
            )
        if len(matched) == 1:
            return next(iter(matched.items()))

        spatial = next(
            (self._SPATIAL_RANK[t] for t in tokens if t in self._SPATIAL_RANK), None,
        )
        if spatial is not None:
            # Rank only actors still in the workspace (a binned bottle drops out); mirrors _ground_spatial.
            cand = list(matched.items())
            try:
                lo, hi = self._workspace_bounds()
                inside = [kv for kv in cand
                          if np.all(np.asarray(kv[1].get_pose().p, float) >= lo)
                          and np.all(np.asarray(kv[1].get_pose().p, float) <= hi)]
                cand = inside or cand
            except Exception:                         # noqa: BLE001 - bounds are a refinement
                pass
            by_x = sorted(cand, key=lambda kv: float(kv[1].get_pose().p[0]))
            return by_x[len(by_x) // 2] if spatial == "mid" else by_x[spatial]
        rank = next(
            (self._SUPERLATIVE_RANK[t] for t in tokens
             if t in self._SUPERLATIVE_RANK), None,
        )
        if rank is None:
            # Category query: first instance in attribute order, deterministically.
            return sorted(matched.items())[0]
        by_volume = sorted(
            matched.items(),
            key=lambda kv: float(np.prod(kv[1].config["scale"])),
        )
        return by_volume[len(by_volume) // 2] if rank == "mid" else by_volume[rank]

    def _actor_region_mask(
        self, actor: Any, world_pts: np.ndarray, margin: float = 0.015
    ) -> np.ndarray:
        """Mask of ``world_pts`` (N,3 world) inside the actor's GT box/ball grown by ``margin`` m."""
        world_pts = np.asarray(world_pts, dtype=np.float64)
        pose = actor.get_pose()
        center = np.asarray(pose.p, dtype=np.float64)
        q = np.asarray(pose.q, dtype=np.float64)
        rot = SciRotation.from_quat([q[1], q[2], q[3], q[0]])
        rel = world_pts - center
        if "extents" in actor.config:
            half = np.asarray(actor.config["extents"], dtype=np.float64) + margin
            local = rel @ rot.as_matrix()  # world -> actor-local (R^T rel)
            return np.all(np.abs(local) <= half, axis=1)
        radii = [
            float(np.linalg.norm(np.asarray(cp, dtype=np.float64)[:3] - center))
            for _, cp in actor.iter_contact_points("list")
        ]
        radius = max([self._ORACLE_MIN_RADIUS, *radii]) + margin
        return np.linalg.norm(rel, axis=1) <= radius

    def _gt_object_points(self, actor: Any) -> np.ndarray:
        """Synthetic GT points: uniform in the annotated box, or a contact-point-radius ball for meshes."""
        pose = actor.get_pose()
        center = np.asarray(pose.p, dtype=np.float64)
        q = np.asarray(pose.q, dtype=np.float64)
        rot = SciRotation.from_quat([q[1], q[2], q[3], q[0]])
        rng = np.random.default_rng(0)
        if "extents" in actor.config:
            half = np.asarray(actor.config["extents"], dtype=np.float64)
            local = rng.uniform(-1.0, 1.0, size=(self._ORACLE_N_POINTS, 3)) * half
            return center + local @ rot.as_matrix().T
        radii = [
            float(np.linalg.norm(np.asarray(cp, dtype=np.float64)[:3] - center))
            for _, cp in actor.iter_contact_points("list")
        ]
        radius = max([self._ORACLE_MIN_RADIUS, *radii])
        d = rng.normal(size=(self._ORACLE_N_POINTS, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        r = radius * rng.uniform(0.0, 1.0, size=(self._ORACLE_N_POINTS, 1)) ** (1 / 3)
        return center + d * r

    # ------------------------- perception overrides -----------------------
    def get_object_3d_points_and_masks_from_language(
        self,
        text_prompt: str,
        use_multiview: bool = True,
    ) -> dict[str, Any]:
        if "perception" not in self._privilege:
            return super().get_object_3d_points_and_masks_from_language(
                text_prompt, use_multiview=use_multiview
            )
        _, actor = self._resolve_actor(text_prompt)

        if "grasp" in self._privilege:
            # No GraspNet downstream: synthetic points stay dense even for a gripper-occluded held object.
            points = np.asarray(self._gt_object_points(actor))
            camera_points = {self.camera_name: points}
            head_pts = points
            points_3d = points
        else:
            # Real GraspNet needs observed surfaces: real points masked to the GT region.
            obs = self.get_observation()
            cameras = [self.camera_name]
            if use_multiview:
                cameras += [c for c in self.aux_camera_names if c in obs]

            camera_points = {}
            for cam_name in cameras:
                depth = obs[cam_name]["images"]["depth"]
                intrinsics = obs[cam_name]["intrinsics"]
                extrinsics = obs[cam_name]["pose_mat"]
                pts_cam = depth_to_pointcloud(depth, intrinsics, subsample_factor=1)
                pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=1)
                pts_world = (extrinsics @ pts_h.T).T[:, :3]
                sel = pts_world[self._actor_region_mask(actor, pts_world)]
                if len(sel) > 0:
                    camera_points[cam_name] = sel

            head_pts = camera_points.get(self.camera_name, np.empty((0, 3)))
            points_3d = (
                np.concatenate(list(camera_points.values()))
                if camera_points else np.empty((0, 3))
            )
            # Full occlusion: fall back to the synthetic cloud.
            if len(points_3d) == 0:
                points_3d = head_pts = np.asarray(self._gt_object_points(actor))
                camera_points = {self.camera_name: points_3d}

        pts_arr = np.asarray(points_3d)
        if pts_arr.size:
            summary = {"centroid": pts_arr.mean(axis=0),
                       "n_points": int(len(pts_arr)),
                       "p5": np.percentile(pts_arr, 5, axis=0),
                       "p95": np.percentile(pts_arr, 95, axis=0)}
        else:
            summary = {"centroid": None, "n_points": 0}
        self._shadow_log("get_object_3d_points_and_masks_from_language",
                         {"text_prompt": text_prompt,
                          "use_multiview": use_multiview}, summary)
        return {
            "head_mask": None,
            "points_3d": np.asarray(points_3d),
            "head_points_3d": np.asarray(head_pts),
            "head_score": 1.0,
            "camera_masks": {c: None for c in camera_points},
            "camera_scores": {c: 1.0 for c in camera_points},
            "camera_points_3d": camera_points,
        }

    def get_object_pose(
        self, object_name: str, use_multiview: bool = True
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        # Same z-down + endpose-frame conversion the perception version applies to its OBB.
        if "perception" not in self._privilege:
            return super().get_object_pose(object_name, use_multiview=use_multiview)
        try:
            _, actor = self._resolve_actor(object_name)
        except ValueError as exc:
            print(f"get_object_pose: '{object_name}' not resolved ({exc})")
            self._shadow_log("get_object_pose",
                             {"object_name": object_name,
                              "use_multiview": use_multiview}, None)
            return None, None
        pose = actor.get_pose()
        position = np.asarray(pose.p, dtype=np.float64)
        q = np.asarray(pose.q, dtype=np.float64)
        R = SciRotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        if R[2, 2] > 0:  # ensure z-axis points down for gripper-frame consistency
            R = R @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        quaternion_wxyz = (
            vtf.SO3.from_matrix(R) @ vtf.SO3(wxyz=self._ZAPPROACH_TO_TCP_WXYZ)
        ).wxyz
        self._shadow_log("get_object_pose",
                         {"object_name": object_name,
                          "use_multiview": use_multiview},
                         {"position": position,
                          "quaternion_wxyz": quaternion_wxyz})
        return position, quaternion_wxyz

    # --------------------------- grasp override ---------------------------
    def _planned_grasp(self, actor: Any, arm_tag: Any, contact_point_id: int) -> np.ndarray | None:
        """Official grasp [xyz, wxyz] at one contact point if plannable for this arm, else None."""
        self.task.plan_success = True  # latch reset: get_grasp_pose early-outs on False
        # get_grasp_pose already vets reachability ([-1]*7); an extra probe would over-reject.
        pose = self.task.get_grasp_pose(
            actor, arm_tag=arm_tag, contact_point_id=contact_point_id, pre_dis=0.0
        )
        if pose is None or np.allclose(pose, [-1] * 7):
            return None
        return np.asarray(pose, dtype=np.float64)

    def _tcp_from_endpose(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(fingertip xyz, quat) of a given 7-vector endpose; pure geometry, unlike the parent's hand-based helper."""
        rot = SciRotation.from_quat([pose[4], pose[5], pose[6], pose[3]])
        return pose[:3] - rot.apply(self._TCP_OFFSET), pose[3:7]

    def _expert_grasp_rank(self, pose: np.ndarray, arm_tag: Any) -> tuple[int, float]:
        """Sort key mirroring Base_Task.choose_grasp_pose: top-down, then preferred side, then the rest."""
        from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC  # type: ignore[import-not-found]
        from envs.utils.transforms import cal_quat_dis  # type: ignore[import-not-found]

        quat = pose[3:7]
        top = "top_down_little_left" if str(arm_tag) == "right" else "top_down_little_right"
        d_top = float(cal_quat_dis(quat, GRASP_DIRECTION_DIC[top]))
        d_side = float(cal_quat_dis(
            quat, GRASP_DIRECTION_DIC[self.task.robot.get_grasp_perfect_direction(arm_tag)]
        ))
        if d_top < 0.15:
            return (0, d_top)
        if d_side < 0.15:
            return (1, d_side)
        return (2, 0.7 * d_top + 0.3 * d_side)

    def sample_grasp_pose(
        self,
        object_name: str,
        use_multiview: bool = True,
        hand: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if "grasp" not in self._privilege:
            return super().sample_grasp_pose(
                object_name, use_multiview=use_multiview, hand=hand
            )
        _, actor = self._resolve_actor(object_name)
        center = np.asarray(actor.get_pose().p, dtype=np.float64)
        if hand in ("left", "right"):
            hands: tuple[str, ...] = (hand,)
        else:
            hands = ("right", "left") if self._profile.near_hand(center) == "right" else ("left", "right")
        # Contact point 0 first; otherwise the rest in the expert's preference order, not file order.
        n_points = len(actor.config.get("contact_points_pose", [])) or 1
        for h in hands:
            arm_tag = self._arm(h)
            first = self._planned_grasp(actor, arm_tag, 0)
            if first is not None:
                return self._tcp_from_endpose(first)
            ranked = []
            for cp in range(1, n_points):
                pose = self._planned_grasp(actor, arm_tag, cp)
                if pose is not None:
                    ranked.append((self._expert_grasp_rank(pose, arm_tag), cp, pose))
            if ranked:
                ranked.sort(key=lambda r: (r[0], r[1]))
                return self._tcp_from_endpose(ranked[0][2])
        raise ValueError(
            f"No reachable grasp for '{object_name}' - call sample_grasp_pose "
            f"again (candidates are stochastic) or try a different object description."
        )


class RoboTwinGripperPrivilegedApi(RoboTwinBasePrivilegedApi, RoboTwinGripperApi):
    """Privileged base + unprivileged gripper by pure composition (no method bodies)."""


__all__ = [
    "RoboTwinBasePrivilegedApi",
    "RoboTwinGripperPrivilegedApi",
]
