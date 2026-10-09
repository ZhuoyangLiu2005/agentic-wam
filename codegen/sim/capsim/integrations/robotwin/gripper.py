"""RoboTwin backend of the :class:`GripperApi` surface (bodies only; docs and prompt order live on the surface)."""

from __future__ import annotations

from typing import Any

import numpy as np
import viser.transforms as vtf

from capsim.integrations.robotwin import task_shims
from capsim.integrations.robotwin.base import RoboTwinBaseApi
from capsim.integrations.surfaces.gripper import GripperApi


class RoboTwinGripperApi(RoboTwinBaseApi, GripperApi):
    """RoboTwin backend of GripperApi (the ASPIRE API)."""

    def functions(self) -> dict[str, Any]:
        return GripperApi.functions(self)

    def open_gripper(self, hand: str) -> bool:
        ok = self._exec(self.task.open_gripper(self._arm(hand)))
        self._note_gripper(hand, closed=False)
        return ok

    def close_gripper(self, hand: str) -> bool:
        ok = self._exec(self.task.close_gripper(self._arm(hand)))
        self._note_gripper(hand, closed=True)
        task_shims.note_close_gripper(self.task, hand)
        return ok

    def get_top_down_grasp_from_obb(
        self,
        obb: dict,
        hand: str | None = None,
        points: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        R = np.asarray(obb["R"], dtype=np.float64)
        extent = np.asarray(obb["extent"], dtype=np.float64)
        position = np.array(obb["center"], dtype=np.float64)  # copy: o3d buffers are read-only
        table_z = float(self.task.table.get_pose().p[2])  # TODO(profile): sim read on the perception tier

        close_dirs: list[np.ndarray] | None = None
        if points is not None and len(points) >= 20:
            # Min-area rectangle of the top 2 cm slab: a partial shell's OBB rotates the jaws onto corners.
            pts = np.asarray(points, dtype=np.float64)
            top_z = float(np.percentile(pts[:, 2], 95))
            slab = pts[pts[:, 2] > top_z - 0.02]
            if len(slab) >= 10:
                from scipy.spatial import ConvexHull

                xy = slab[:, :2]
                hull_pts = xy[ConvexHull(xy).vertices]
                best: tuple | None = None
                for i in range(len(hull_pts)):
                    edge = hull_pts[(i + 1) % len(hull_pts)] - hull_pts[i]
                    theta = np.arctan2(edge[1], edge[0])
                    c, s = np.cos(-theta), np.sin(-theta)
                    rot2d = np.array([[c, -s], [s, c]])
                    r = hull_pts @ rot2d.T
                    w = np.ptp(r[:, 0])
                    h = np.ptp(r[:, 1])
                    if best is None or w * h < best[0]:
                        mid = np.array([r[:, 0].min() + w / 2, r[:, 1].min() + h / 2])
                        best = (w * h, theta, w, h, rot2d.T @ mid)
                _, theta, w, h, center_xy = best
                position[0], position[1] = center_xy
                u = np.array([np.cos(theta), np.sin(theta), 0.0])
                v = np.array([-np.sin(theta), np.cos(theta), 0.0])
                close_dirs = [v, u] if w <= h else [u, v]
                position[2] = max(table_z + 0.5 * (top_z - table_z), table_z + 0.015)

        if close_dirs is None:
            # OBB fallback; a top-heavy partial cloud biases the centre up, so pinch mid-way table-to-top.
            vertical_half = 0.5 * float(np.max(np.abs(R[2, :]) * extent))
            cloud_top = position[2] + vertical_half
            position[2] = max(table_z + 0.5 * (cloud_top - table_z), table_z + 0.015)
            vertical = int(np.argmax(np.abs(R[2, :])))
            horizontal = [i for i in range(3) if i != vertical]
            horizontal.sort(key=lambda i: extent[i])  # narrowest width first
            close_dirs = []
            for i in horizontal:
                d = R[:, i].copy()
                d[2] = 0.0
                n = np.linalg.norm(d)
                close_dirs.append(d / n if n > 1e-6 else np.array([0.0, 1.0, 0.0]))

        def _down_quat(close_dir: np.ndarray) -> np.ndarray:
            x_ee = np.array([0.0, 0.0, -1.0])  # approach straight down
            y_ee = close_dir  # jaws close across this direction
            z_ee = np.cross(x_ee, y_ee)
            return vtf.SO3.from_matrix(np.column_stack([x_ee, y_ee, z_ee])).wxyz

        quats = [_down_quat(d) for d in close_dirs]
        for quat in quats:
            if self._grasp_plan_feasible(position, quat, hand=hand):
                return position, np.asarray(quat, dtype=np.float64)
        # Nothing plannable: return the narrow-width pose so goto_pose surfaces the failure.
        return position, np.asarray(quats[0], dtype=np.float64)


__all__ = ["RoboTwinGripperApi"]
