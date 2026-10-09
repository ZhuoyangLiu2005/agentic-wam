"""Depth -> point-cloud helpers (camera intrinsics/extrinsics to world points)."""
import numpy as np
import numpy.typing as npt
from matplotlib import cm


def deproject_pixel_to_camera(
    pixel: tuple[int, int],
    depth: float,
    intrinsics: npt.NDArray[np.float64],
) -> np.ndarray:
    """Deproject pixel (x, y) at the given depth to a camera-frame point."""
    x = (pixel[0] - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y = (pixel[1] - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    z = depth
    return np.array([x, y, z])


def depth_color_to_pointcloud(
    depth: npt.NDArray[np.float64],
    img: npt.NDArray[np.uint8],
    intrinsics: npt.NDArray[np.float64],
    subsample_factor: int = 1,
    depth_clip_range: tuple[float, float] = (0.015, 20.0),
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Back-project depth + RGB to camera-frame (points, colors), dropping NaN/inf and depth outside depth_clip_range."""
    if len(depth.shape) != 2:
        raise ValueError(f"Depth array must be 2D, got shape {depth.shape}")

    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"Image array must be (H, W, 3), got shape {img.shape}")

    if depth.shape[:2] != img.shape[:2]:
        raise ValueError(
            f"Depth and image dimensions must match: {depth.shape[:2]} vs {img.shape[:2]}"
        )

    if intrinsics.shape != (3, 3):
        raise ValueError(f"Intrinsics must be (3, 3), got shape {intrinsics.shape}")

    if subsample_factor <= 0:
        raise ValueError(f"Subsample factor must be positive, got {subsample_factor}")
    H, W = depth.shape
    H_subsampled = H // subsample_factor
    W_subsampled = W // subsample_factor
    depth = depth[::subsample_factor, ::subsample_factor]
    img = img[::subsample_factor, ::subsample_factor]

    intrinsics = intrinsics.copy()
    intrinsics[0, 0] /= subsample_factor  # fx
    intrinsics[1, 1] /= subsample_factor  # fy
    intrinsics[0, 2] /= subsample_factor  # cx
    intrinsics[1, 2] /= subsample_factor  # cy

    w, h = np.meshgrid(np.arange(W_subsampled), np.arange(H_subsampled), indexing="xy")
    pixels = np.stack([w.flatten(), h.flatten()], axis=-1).astype(np.float32)

    z = depth.reshape(-1)

    x = (pixels[:, 0] - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (pixels[:, 1] - intrinsics[1, 2]) * z / intrinsics[1, 1]

    points = np.stack([x, y, z], axis=-1)

    colors = img.reshape(-1, img.shape[-1])[:, :3] / 255.0

    near_clip, far_clip = depth_clip_range
    valid_mask = (
        ~np.isnan(points).any(axis=1)
        & ~np.isinf(points).any(axis=1)
        & (points[:, 2] <= far_clip)
        & (points[:, 2] >= near_clip)
    )

    return points[valid_mask], colors[valid_mask]


def depth_to_pointcloud(
    depth: npt.NDArray[np.float64],
    intrinsics: npt.NDArray[np.float64],
    subsample_factor: int = 1,
    depth_clip_range: tuple[float, float] = (0.015, 20.0),
    filter_invalid: bool = True,
) -> npt.NDArray[np.float64]:
    """Back-project depth to an (N, 3) camera-frame point cloud; filter_invalid=False keeps one point per pixel (1:1 with the subsampled image)."""
    if len(depth.shape) != 2:
        raise ValueError(f"Depth array must be 2D, got shape {depth.shape}")

    if intrinsics.shape != (3, 3):
        raise ValueError(f"Intrinsics must be (3, 3), got shape {intrinsics.shape}")

    if subsample_factor <= 0:
        raise ValueError(f"Subsample factor must be positive, got {subsample_factor}")
    H, W = depth.shape
    H_subsampled = H // subsample_factor
    W_subsampled = W // subsample_factor
    depth = depth[::subsample_factor, ::subsample_factor]

    intrinsics = intrinsics.copy()
    intrinsics[0, 0] /= subsample_factor  # fx
    intrinsics[1, 1] /= subsample_factor  # fy
    intrinsics[0, 2] /= subsample_factor  # cx
    intrinsics[1, 2] /= subsample_factor  # cy

    w, h = np.meshgrid(np.arange(W_subsampled), np.arange(H_subsampled), indexing="xy")
    pixels = np.stack([w.flatten(), h.flatten()], axis=-1).astype(np.float32)

    z = depth.reshape(-1)
    x = (pixels[:, 0] - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (pixels[:, 1] - intrinsics[1, 2]) * z / intrinsics[1, 1]

    points = np.stack([x, y, z], axis=-1)

    if filter_invalid:
        near_clip, far_clip = depth_clip_range
        valid_mask = (
            ~np.isnan(points).any(axis=1)
            & ~np.isinf(points).any(axis=1)
            & (points[:, 2] <= far_clip)
            & (points[:, 2] >= near_clip)
        )
        return points[valid_mask]

    return points


def depth_to_rgb(
    depth: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap_name: str = "viridis",
    invalid_is_nan: bool = True,
    invalid_color: tuple[int, int, int] = (0, 0, 0),
    use_percentiles: tuple[float, float] | None = None,  # e.g., (2, 98)
    invert: bool = False,
    log_scale: bool = False,
) -> np.ndarray:
    """Colormap a (H, W) depth map to a uint8 RGB image; invalid (NaN/inf) pixels get invalid_color."""
    assert depth.ndim == 2, "depth must be (H, W)"
    d = depth.astype(np.float64).copy()

    invalid_mask = (np.isnan(d) | np.isinf(d)) if invalid_is_nan else np.zeros_like(d, dtype=bool)

    if log_scale:
        min_pos = np.nanmin(d[d > 0]) if np.any(d > 0) else 1.0
        d = np.log1p(np.maximum(d, min_pos * 1e-9))

    valid = d[~invalid_mask]
    if valid.size == 0:
        h, w = d.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[...] = np.array(invalid_color, dtype=np.uint8)
        return rgb

    if use_percentiles is not None:
        low_p, high_p = use_percentiles
        vmin = np.percentile(valid, low_p) if vmin is None else vmin
        vmax = np.percentile(valid, high_p) if vmax is None else vmax
    else:
        vmin = np.min(valid) if vmin is None else vmin
        vmax = np.max(valid) if vmax is None else vmax

    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = (d - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)

    if invert:
        norm = 1.0 - norm

    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm)  # (H, W, 4)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)

    if invalid_is_nan:
        rgb[invalid_mask] = np.array(invalid_color, dtype=np.uint8)

    return rgb
