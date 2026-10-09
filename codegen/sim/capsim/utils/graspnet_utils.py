"""Request/response helpers shared by the GraspNet server and client."""
import base64
import io

import numpy as np
import viser.transforms as vtf
from fastapi import HTTPException


def _base64_to_numpy(b64_str: str) -> np.ndarray:
    try:
        data = base64.b64decode(b64_str)
        with io.BytesIO(data) as f:
            return np.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid numpy data: {e}")


def _numpy_to_base64(arr: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, arr)
        return base64.b64encode(f.getvalue()).decode("utf-8")


def camera_so3_looking_at_origin(cam_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R_cw, R_wc) for a camera at cam_pos looking at the origin, image +y along world +z."""
    c = np.asarray(cam_pos, dtype=float)

    z_c = -c
    z_c /= np.linalg.norm(z_c)

    down_world = np.array([0.0, 0.0, 1.0])

    y_c = down_world - np.dot(down_world, z_c) * z_c
    y_c /= np.linalg.norm(y_c)

    x_c = np.cross(y_c, z_c)

    R_wc = np.column_stack([x_c, y_c, z_c])

    R_cw = R_wc.T
    return R_cw, R_wc


def calculate_look_at_rotation(position: np.ndarray, target_point: np.ndarray) -> np.ndarray:
    """Camera orientation (wxyz) looking at target_point, OpenCV convention (Z forward, Y down, X right)."""
    z_c = target_point - position
    z_c_norm = np.linalg.norm(z_c)
    if z_c_norm > 1e-6:
        z_c /= z_c_norm
    else:
        z_c = np.array([0.0, 0.0, 1.0])  # Fallback

    # heuristic: +Y is down
    down_world = np.array([0.0, 1.0, 0.0])

    y_c = down_world - np.dot(down_world, z_c) * z_c
    y_c_norm = np.linalg.norm(y_c)
    if y_c_norm < 1e-6:
        # degenerate (looking straight down/up): use X as the temporary down
        y_c = np.array([1.0, 0.0, 0.0])
        y_c = y_c - np.dot(y_c, z_c) * z_c
        y_c /= np.linalg.norm(y_c)
    else:
        y_c /= y_c_norm

    x_c = np.cross(y_c, z_c)

    # columns are camera axes expressed in world
    R_wc = np.column_stack([x_c, y_c, z_c])

    wxyz = vtf.SO3.from_matrix(R_wc).wxyz
    return wxyz


def sample_hemisphere_viewpoint(
    target_point: np.ndarray,
    current_camera_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one random viewpoint on the hemisphere around target_point whose pole is the current camera; returns (position, wxyz)."""
    vec_cam_target = current_camera_position - target_point
    radius = np.linalg.norm(vec_cam_target)
    if radius < 1e-6:
        radius = 0.5
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        z_axis = vec_cam_target / radius

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, z_axis)) > 0.99:
        helper = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # uniform on the sphere surface = uniform cos(theta); hemisphere: cos(theta) in [0, 1]
    z_local = np.random.uniform(0.0, 1.0)
    phi = np.random.uniform(0.0, 2 * np.pi)

    sin_theta = np.sqrt(1 - z_local**2)
    x_local = sin_theta * np.cos(phi)
    y_local = sin_theta * np.sin(phi)

    offset = x_local * x_axis + y_local * y_axis + z_local * z_axis
    position = target_point + radius * offset

    wxyz = calculate_look_at_rotation(position, target_point)

    return position, wxyz


def sample_hemisphere_viewpoints_evenly(
    target_point: np.ndarray,
    current_camera_position: np.ndarray,
    num_samples: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample num_samples hemisphere viewpoints on a Fibonacci lattice; returns (positions (N,3), wxyzs (N,4))."""
    vec_cam_target = current_camera_position - target_point
    radius = np.linalg.norm(vec_cam_target)
    if radius < 1e-6:
        radius = 0.5
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        z_axis = vec_cam_target / radius

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, z_axis)) > 0.99:
        helper = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    positions = []
    wxyzs = []

    phi = (np.sqrt(5) + 1) / 2

    for i in range(num_samples):
        # z from 1 (pole) down to 0 (equator)
        if num_samples > 1:
            z_local = 1 - (i / (num_samples - 1))
        else:
            z_local = 1.0

        theta = 2 * np.pi * i / phi

        sin_theta = np.sqrt(1 - z_local**2)
        x_local = sin_theta * np.cos(theta)
        y_local = sin_theta * np.sin(theta)

        offset = x_local * x_axis + y_local * y_axis + z_local * z_axis
        position = target_point + radius * offset

        positions.append(position)
        wxyzs.append(calculate_look_at_rotation(position, target_point))

    # keep the two thirds of the viewpoints closest to the world origin
    furthest_indices = np.argsort(np.linalg.norm(np.array(positions), axis=1))[
        : len(positions) * 2 // 3
    ]
    positions = np.array(positions)[furthest_indices]
    wxyzs = np.array(wxyzs)[furthest_indices]

    return np.array(positions), np.array(wxyzs)


def sample_cone_viewpoints_evenly(
    target_point: np.ndarray,
    current_camera_position: np.ndarray,
    num_samples: int = 100,
    max_angle_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Like sample_hemisphere_viewpoints_evenly but on a spherical cap of half-angle max_angle_deg (90 = hemisphere)."""
    vec_cam_target = current_camera_position - target_point
    radius = np.linalg.norm(vec_cam_target)
    if radius < 1e-6:
        radius = 0.5
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        z_axis = vec_cam_target / radius

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, z_axis)) > 0.99:
        helper = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    positions = []
    wxyzs = []

    # on a cap z_local spans [cos(max_angle), 1] instead of [0, 1]
    max_angle_rad = np.radians(max_angle_deg)
    min_z = np.cos(max_angle_rad)

    phi = (np.sqrt(5) + 1) / 2

    for i in range(num_samples):
        if num_samples > 1:
            t = i / (num_samples - 1)
        else:
            t = 0.0

        # linear in z = uniform area on the cap
        z_local = 1 - t * (1 - min_z)

        theta = 2 * np.pi * i / phi

        sin_theta = np.sqrt(1 - z_local**2)
        x_local = sin_theta * np.cos(theta)
        y_local = sin_theta * np.sin(theta)

        offset = x_local * x_axis + y_local * y_axis + z_local * z_axis
        position = target_point + radius * offset

        positions.append(position)
        wxyzs.append(calculate_look_at_rotation(position, target_point))

    return np.array(positions), np.array(wxyzs)


def sample_random_camera_viewpoint(
    target_point: np.ndarray, xy_extent_meters: float = 0.25
) -> tuple[np.ndarray, np.ndarray]:
    # random position within xy_extent_meters of the origin, looking at target_point; returns (position, wxyz)
    position = np.random.uniform(-xy_extent_meters, xy_extent_meters, 3)

    z_c = target_point - position
    z_c /= np.linalg.norm(z_c)

    # Any stable reference vector works; in the camera convention (x-right, y-down, z-forward) "down" is +y.
    down_world = np.array([0.0, 1.0, 0.0])

    y_c = down_world - np.dot(down_world, z_c) * z_c
    y_c_norm = np.linalg.norm(y_c)
    if y_c_norm < 1e-6:
        y_c = np.array([1.0, 0.0, 0.0])
        y_c = y_c - np.dot(y_c, z_c) * z_c
        y_c /= np.linalg.norm(y_c)
    else:
        y_c /= y_c_norm

    x_c = np.cross(y_c, z_c)

    # columns are camera axes expressed in world
    R_wc = np.column_stack([x_c, y_c, z_c])

    wxyz = vtf.SO3.from_matrix(R_wc).wxyz
    return position, wxyz
