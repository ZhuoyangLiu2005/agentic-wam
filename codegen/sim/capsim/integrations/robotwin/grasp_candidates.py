"""Perception-tier grasp candidate geometry (pure numpy), ordered by :func:`propose` for ``sample_grasp_pose``.

Poses are endpose-frame fingertip poses: position = TCP, quat wxyz with x = approach, y = jaw-closing line.
"""
from __future__ import annotations
import re
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from capsim.integrations.robotwin import vocab as _vocab
from capsim.profiles import DEFAULT_EMBODIMENT, load_profile

# Grasp-family hints keyed by the words of the name (object_vocab.yaml `grasp`).
_GRASP = _vocab.section("grasp")
RIM_RE = _vocab.words_re(_GRASP["rim_words"], re.I)
TOP_EDGE_RE = _vocab.words_re(_GRASP["top_edge_words"], re.I)   # taken by the top edge: fingertips ~1.5 cm below the top
TALL_RE = _vocab.words_re(_GRASP["tall_words"], re.I)
PANEL_RE = _vocab.words_re(_GRASP["panel_words"], re.I)
_POL = _vocab.policy()["grasp"]      # tuning values: grounding_policy.yaml `grasp`
W_JD = _POL["jaw_distance_weight"]
SCORE_ORDER = bool(_POL["score_order"])
FLAT_H = _POL["flat_height"]
GATE = _POL["off_centre_gate"]


def base_noun(noun):
    """'bottles3' -> 'bottles', 'bottle_2' -> 'bottle': programs index instances with a trailing number."""
    return re.sub(r"[\s_]*\d+$", "", (noun or "").strip().lower())


def quat_to_R(q_wxyz):
    q = np.asarray(q_wxyz, float)
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def R_to_quat(M):
    x, y, z, w = R.from_matrix(M).as_quat()
    return np.array([w, x, y, z])


def frame_quat(approach, jaw):
    """endpose-frame quat: x = approach, y = jaw closing line, z = x cross y."""
    x = np.asarray(approach, float); x = x / np.linalg.norm(x)
    y = np.asarray(jaw, float); y = y - x * (y @ x); y = y / np.linalg.norm(y)
    return R_to_quat(np.column_stack([x, y, np.cross(x, y)]))


def min_area_rect(pts_xy):
    P = np.asarray(pts_xy, float)
    if len(P) > 3000:
        P = P[:: (len(P) // 3000) + 1]
    try:
        H = P[ConvexHull(P).vertices]
    except Exception:
        return None
    best = None
    for i in range(len(H)):
        e = H[(i + 1) % len(H)] - H[i]
        if np.hypot(*e) < 1e-9:
            continue
        th = float(np.arctan2(e[1], e[0]))
        c, s = np.cos(-th), np.sin(-th)
        M = np.array([[c, -s], [s, c]])
        Q = H @ M.T
        w = float(np.ptp(Q[:, 0])); l = float(np.ptp(Q[:, 1]))
        if best is None or w * l < best["area"]:
            mid = np.array([(Q[:, 0].min() + Q[:, 0].max()) / 2, (Q[:, 1].min() + Q[:, 1].max()) / 2])
            best = {"area": w * l, "yaw": th, "w": w, "l": l, "centre": M.T @ mid}
    return best


def object_geometry(cloud, table_z):
    """base_z, height, footprint rect (centre/w/l/yaw, w <= l), centroid, top_z."""
    a = np.asarray(cloud, float)
    z_top = float(np.percentile(a[:, 2], 97))
    z_bot = float(np.percentile(a[:, 2], 3))
    resting = (z_bot - table_z) < 0.03 or table_z is None
    base_z = table_z if resting else z_bot
    rect = min_area_rect(a[:, :2])
    if rect is None:
        c = a[:, :2].mean(0); ext = np.percentile(a[:, :2], 98, 0) - np.percentile(a[:, :2], 2, 0)
        rect = {"centre": c, "w": float(min(ext)), "l": float(max(ext)), "yaw": 0.0, "area": float(ext[0] * ext[1])}
    if rect["w"] > rect["l"]:
        rect["w"], rect["l"] = rect["l"], rect["w"]; rect["yaw"] += np.pi / 2
    return {"base_z": base_z, "top_z": z_top, "h": max(z_top - base_z, 0.005), "centre": rect["centre"],
            "w": rect["w"], "l": rect["l"], "yaw": rect["yaw"], "centroid": a.mean(0), "resting": resting}


def is_tall(geom, noun):
    """A standing cylinder-like object: taller than its footprint and round-ish, or a standing bottle/can noun."""
    w, l, h = geom["w"], geom["l"], geom["h"]
    standing = h > 1.5 * w and h >= 0.8 * l      # a lying bottle / hammer is long, not tall
    roundish = l <= 1.6 * w
    return standing and (roundish and h > 2.5 * w or bool(TALL_RE.search(base_noun(noun))))


def jaw_line_distance(pos, quat, centre_xy):
    """xy distance from the footprint centre to the jaw-closing line through the fingertip."""
    jaw = quat_to_R(quat)[:, 1][:2]
    n = np.linalg.norm(jaw)
    if n < 1e-6:
        return float(np.linalg.norm(pos[:2] - centre_xy))
    jaw = jaw / n
    d = pos[:2] - np.asarray(centre_xy)
    return float(abs(d[0] * jaw[1] - d[1] * jaw[0]))


def top_down_z_ref(geom, noun):
    """Top-down closing height: 1.5 cm below the top for rim/top-edge objects, else 2 cm (>= mid-height)."""
    base, h = geom["base_z"], geom["h"]
    if TOP_EDGE_RE.search(noun or ""):
        return base + max(h - 0.015, 0.5 * h)
    return base + max(min(h - 0.02, 0.9 * h), 0.5 * h)


def robot_direction(robot_dir=None):
    """`robot_dir` as a float array, else the default profile's robot.base_dir."""
    if robot_dir is None:
        robot_dir = load_profile(DEFAULT_EMBODIMENT).robot.base_dir
    return np.asarray(robot_dir, dtype=np.float64)


def filter_and_rank(cands, geom, noun, table_z, allow_side=True, robot_dir=None):
    """Filter (pos, quat_wxyz, score) candidates and return (pos, quat, score, tags), best first."""
    robot_dir = robot_direction(robot_dir)
    h = geom["h"]; base = geom["base_z"]
    z_min = table_z + (0.005 if h < 0.025 else 0.010)
    rim = bool(RIM_RE.search(noun or ""))
    compact = max(geom["w"], geom["l"]) < 0.20 and not rim and geom["l"] <= 2.5 * geom["w"]   # elongated tools are held by the handle: off-centre is legitimate
    kept = []
    for pos, quat, score in cands:
        pos = np.asarray(pos, float); app = quat_to_R(quat)[:, 0]
        tags = []
        if pos[2] < z_min:
            continue                                   # fingers into the table
        if app[2] > 0.3:
            continue                                   # approaching from below
        if app @ robot_dir > 0.5:
            continue                                   # from beyond the far table edge, toward the robot
        if not allow_side and abs(app[2]) < 0.5:
            continue
        if geom["h"] < FLAT_H and abs(app[2]) < 0.5:
            continue                                   # too thin to pinch from the side
        jd = jaw_line_distance(pos, quat, geom["centre"])
        if compact and jd > GATE * max(geom["w"], geom["l"]):
            continue                                   # pinching an edge / corner (or the thin end of a lying bottle)
        top_down = app[2] < -0.7
        if top_down:
            key = abs(pos[2] - top_down_z_ref(geom, noun)) + W_JD * jd   # near the ideal height AND through the footprint centre
            tags.append("top")
        else:
            key = abs(pos[2] - (base + 0.5 * h)) + W_JD * jd
            tags.append("side")
        kept.append((key, -float(score), pos, np.asarray(quat, float), float(score), tags, jd))
    kept.sort(key=lambda k: (k[1],) if SCORE_ORDER else (k[0], k[1]))
    return [(k[2], k[3], k[4], k[5] + [f"jd={k[6]:.3f}"]) for k in kept]


def side_pinch_family(geom, noun, table_z, yaws_deg=(0, -30, 30, -60, 60, -90, 90)):
    """Mid-height body pinches for tall objects from the robot's side, jaws horizontal."""
    c = np.asarray(geom["centre"], float); h = geom["h"]; base = geom["base_z"]
    r = 0.5 * min(geom["w"], geom["l"])
    z = max(base + 0.5 * h, table_z + 0.02)
    out = []
    for i, yd in enumerate(yaws_deg):
        th = np.deg2rad(yd)
        toward = np.array([np.sin(th), np.cos(th), 0.0])   # +y = away from the robot
        fingertip = np.array([c[0], c[1], z]) - toward * (r + 0.0)   # fingertips at the surface on the near side
        jaw = np.array([np.cos(th), -np.sin(th), 0.0])      # horizontal, perpendicular to the approach
        out.append((fingertip, frame_quat(toward, jaw), 1.0 - 0.05 * i, ["side-family", f"yaw={yd}"]))
    return out


def top_down_fallback(geom, table_z, noun=""):
    """Deterministic pinch across the SHORT side of the footprint rectangle."""
    c = np.asarray(geom["centre"], float); h = geom["h"]; base = geom["base_z"]
    z = float(np.clip(top_down_z_ref(geom, noun), table_z + 0.01, base + h - 0.005))
    yaw = geom["yaw"]                                   # direction of the footprint's short (w) side
    jaw = np.array([np.cos(yaw), np.sin(yaw), 0.0])     # jaws close along the short (w) side
    return [(np.array([c[0], c[1], z]), frame_quat([0, 0, -1.0], jaw), 0.5, ["fallback-topdown"])]


PANEL_RMS = _POL["panel_rms"]
PANEL_TILT = tuple(_POL["panel_tilt_deg"])
PANEL_SPAN = _POL["panel_span"]
PANEL_INSET = _POL["panel_inset"]


def panel_edge_family(cloud, geom, table_z, noun="", n_along=3):
    """Pinch a tilted panel's (laptop lid's) free edge, approach down-slope; [] when no such panel is seen."""
    if cloud is None or not PANEL_RE.search(base_noun(noun) or ""):
        return []
    a = np.asarray(cloud, float)
    if len(a) < 120:
        return []
    h, top = geom["h"], geom["top_z"]
    if not 0.03 <= h <= 0.25:
        return []
    band = a[a[:, 2] > top - max(0.6 * h, 0.02)]      # the panel: the upper slab of the cloud
    if len(band) < 40:
        return []
    c = band.mean(0)
    try:
        s_val, vt = np.linalg.svd(band - c, full_matrices=False)[1:]
    except np.linalg.LinAlgError:
        return []
    n = vt[2] if vt[2][2] > 0 else -vt[2]             # panel normal, pointing up
    rms = float(s_val[2] / np.sqrt(len(band)))
    tilt = float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0))))
    if rms > PANEL_RMS or not PANEL_TILT[0] <= tilt <= PANEL_TILT[1]:
        return []                                     # not a thin, tilted panel
    down = np.array([0.0, 0.0, 1.0]) - n * n[2]       # steepest direction inside the panel
    nd = np.linalg.norm(down)
    if nd < 1e-6:
        return []
    down = -down / nd                                 # down-slope = toward the hinge
    edge_dir = np.cross(n, down)                      # along the free edge
    t = band @ (-down)
    edge = band[t > t.max() - 0.010]                  # the free edge: the up-slope extreme
    if len(edge) < 10:
        return []
    span = float(np.ptp(edge @ edge_dir))
    if span < PANEL_SPAN:
        return []
    mid = edge.mean(0)
    quat = frame_quat(down, n)                        # x = approach (down-slope), y = jaws (panel normal)
    offs = [0.0] + [o for k in range(1, n_along) for o in (k * 0.25 * span, -k * 0.25 * span)]
    out = []
    for i, off in enumerate(offs[:n_along]):
        pos = mid + down * PANEL_INSET - n * 0.003 + edge_dir * off
        out.append((pos, quat, 1.0 - 0.01 * i, ["panel-edge", f"tilt={tilt:.0f}", f"off={off*100:.0f}cm"]))
    return out


def propose(cands, geom, noun, table_z, cloud=None, robot_dir=None):
    """Full ordering: panel edge, tall-object side family, ranked GraspNet (caps last), top-down fallback."""
    noun = base_noun(noun)
    panel = panel_edge_family(cloud, geom, table_z, noun) if cloud is not None else []
    ranked = filter_and_rank(cands, geom, noun, table_z, robot_dir=robot_dir)
    tall = is_tall(geom, noun)
    if tall:
        body = [k for k in ranked if k[0][2] <= geom["base_z"] + 0.8 * geom["h"]]
        cap = [k for k in ranked if k[0][2] > geom["base_z"] + 0.8 * geom["h"]]
        ordered = side_pinch_family(geom, noun, table_z) + body + cap
    else:
        ordered = ranked
    ordered = panel + ordered + top_down_fallback(geom, table_z, noun)
    if not tall:
        # Last resort: horizontal contacts for push actions (switch, button), never changing the first proposal.
        ordered = ordered + [(p, q, s, ["fallback-side"] + t[1:])
                             for p, q, s, t in side_pinch_family(geom, noun, table_z, yaws_deg=(0, -45, 45, -90, 90))]
    return ordered
