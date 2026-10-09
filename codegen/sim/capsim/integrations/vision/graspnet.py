"""Client for the Contact-GraspNet grasp-synthesis server (capsim/serving/launch_contact_graspnet_server.py)."""
from __future__ import annotations

import base64
import io
import os
from typing import Any

import numpy as np
import requests

SERVICE_URL = os.environ.get("GRASPNET_SERVICE_URL", "http://127.0.0.1:8115")


# --- Serialization Helpers ---

def _numpy_to_base64(arr: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, arr)
        return base64.b64encode(f.getvalue()).decode("utf-8")


def _base64_to_numpy(b64_str: str) -> np.ndarray:
    try:
        data = base64.b64decode(b64_str)
        with io.BytesIO(data) as f:
            return np.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to decode numpy data: {e}")


def init_contact_graspnet_point_clouds() -> Any:
    """Initialize a GraspNet client that plans grasps from pre-computed point clouds."""

    def plan_point_clouds(
        pc_full: np.ndarray,
        pc_segment: np.ndarray,
        segmap_id: int = 1,
        local_regions: bool = True,
        filter_grasps: bool = True,
        forward_passes: int = 2,
        max_retries: int = 10,
    ) -> tuple[np.ndarray | list, np.ndarray | list, np.ndarray | list]:
        """Call the GraspNet service with pre-computed point clouds."""
        payload = {
            "pc_full_base64": _numpy_to_base64(pc_full),
            "pc_segment_base64": _numpy_to_base64(pc_segment),
            "segmap_id": segmap_id,
            "local_regions": local_regions,
            "filter_grasps": filter_grasps,
            "forward_passes": forward_passes,
            "max_retries": max_retries,
        }

        try:
            resp = requests.post(f"{SERVICE_URL}/plan_point_clouds", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to communicate with GraspNet service at {SERVICE_URL}: {e}"
            )

        grasps = _base64_to_numpy(data["grasps_base64"])
        scores = _base64_to_numpy(data["scores_base64"])
        contact_pts = _base64_to_numpy(data["contact_pts_base64"])

        return grasps, scores, contact_pts

    return plan_point_clouds


