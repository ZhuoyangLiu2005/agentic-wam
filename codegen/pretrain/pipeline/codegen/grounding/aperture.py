"""Hand contact-event detection for EGOCENTRIC human episodes — the ego
analog of ``transitions.grip_transitions``.

Human data has no gripper sensor, so the observable proxy for hand-object
contact is the Qwen-RoboManip retargeted gripper width: the distance from the
thumb tip to a virtual finger blending the index and middle fingertips,

    k_vf = 0.7 * k_index + 0.3 * k_middle,   w(t) = ||k_thumb - k_vf||,

Savitzky-Golay smoothed (~0.5 s window). The width is a DISTANCE between
points of the same rigid-ish hand, so it is invariant to the moving egocentric
camera — camera-frame keypoints are exactly as good as world-frame here.

Episode-relative thresholds only (hand scale and retargeting differ per source
dataset — never absolute meters): hysteresis between the 35% and 60% points of
the episode's own 5th..95th percentile range; a hand whose range is degenerate
(< 1 cm) emits NO events (idle hand); states shorter than ``min_dwell`` frames
merge away (incidental flexes); a LEADING open (rest-pose uncurl during the
reach) is dropped; an "open" phase that never reaches halfway to the episode's
TRUE open extreme (99.5th pct) is finger SPLAY under load, not a release — it
is dropped together with its re-close.

This proxy is honest but weak — a pinch around empty air looks identical to a
grasp — so downstream reconciliation treats a mismatch as diagnostic
(block-level alignment fallback), never as an error. Dropped events are
returned separately so the prompt/diagnostics can mention them.
"""
from __future__ import annotations

import re

import numpy as np

#: Subtask labels that IMPLY a hand-contact change — the human annotation is
#: ground truth, so a matching label licenses a second, label-conditioned
#: detector pass with WINDOW-relative thresholds (a small pinch drowns in
#: episode-relative thresholds when the episode also contains large grasps).
CONTACT_LABEL = re.compile(
    r"\b(pick|grasp|grab|place|put|press|zip|unzip|insert|hold|take|pull|open|"
    r"close|hang|pour|screw|unscrew|plug|attach|remove|lift)\b", re.I)


def _smooth(x: np.ndarray, k: int = 7) -> np.ndarray:
    return np.convolve(x, np.ones(k) / k, mode="same")


def retargeted_width(thumb_pos: np.ndarray, index_pos: np.ndarray,
                     middle_pos: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Qwen-RoboManip retargeted gripper width w(t) for one hand, SavGol-smoothed."""
    from scipy.signal import savgol_filter
    kvf = 0.7 * np.asarray(index_pos) + 0.3 * np.asarray(middle_pos)
    w = np.linalg.norm(np.asarray(thumb_pos) - kvf, axis=1)
    wl = max(int(fps * 0.5) | 1, 5)          # odd, ~0.5 s
    if len(w) <= wl:
        return _smooth(w)
    return savgol_filter(w, window_length=wl, polyorder=3)


def aperture_events(
    thumb_pos: np.ndarray,
    index_pos: np.ndarray,
    middle_pos: np.ndarray,
    fps: float = 30.0,
    min_dwell: int = 20,
    min_range: float = 0.01,
    drop_leading_open: bool = True,
):
    """Ordered contact transitions of ONE hand from fingertip positions (T, 3).

    Returns (events, dropped): events = [(frame, "close"|"open")], dropped = the
    suppressed events [(frame, kind-with-reason)] (possibly empty)."""
    a = retargeted_width(thumb_pos, index_pos, middle_pos, fps)
    lo, hi = np.percentile(a, 5), np.percentile(a, 95)
    if hi - lo < min_range:
        return [], []
    t_close, t_open = lo + 0.35 * (hi - lo), lo + 0.6 * (hi - lo)

    segs = [[0, "closed" if a[0] < t_close else "open"]]
    for i in range(1, len(a)):
        if segs[-1][1] == "open" and a[i] < t_close:
            segs.append([i, "closed"])
        elif segs[-1][1] == "closed" and a[i] > t_open:
            segs.append([i, "open"])

    # min-dwell: repeatedly remove the first too-short interior state, re-merging
    # equal neighbors, so blips vanish instead of generating event pairs.
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for j in range(1, len(segs)):
            end = segs[j + 1][0] if j + 1 < len(segs) else len(a)
            if end - segs[j][0] < min_dwell:
                segs.pop(j)
                changed = True
                break
        k = 1
        while k < len(segs):
            if segs[k][1] == segs[k - 1][1]:
                segs.pop(k)
            else:
                k += 1

    events = [(int(f), "close" if s == "closed" else "open") for f, s in segs[1:]]
    dropped: list[tuple[int, str]] = []
    if drop_leading_open and events and events[0][1] == "open":
        dropped.append((events[0][0], "open(leading)"))
        events.pop(0)

    # splay suppression: an "open" phase that never reaches halfway to the episode's
    # true open extreme is a re-grip under load, not a release. Drop the open AND the
    # re-close that ends it (a trailing open with a wide peak survives).
    t_true_open = lo + 0.5 * (np.percentile(a, 99.5) - lo)
    kept: list[tuple[int, str]] = []
    j = 0
    while j < len(events):
        f, kind = events[j]
        if kind == "open":
            end = events[j + 1][0] if j + 1 < len(events) else len(a)
            if a[f:end].max() < t_true_open:
                dropped.append((f, "open(splay)"))
                if j + 1 < len(events) and events[j + 1][1] == "close":
                    dropped.append((events[j + 1][0], "close(splay)"))
                    j += 2
                else:
                    j += 1
                continue
        kept.append((f, kind))
        j += 1
    return kept, dropped


def label_boosted_events(thumb_pos, index_pos, middle_pos, base_events, segments,
                         fps: float = 30.0, pad: int | None = None,
                         min_dwell: int = 10, min_range: float = 0.005):
    """Label-conditioned SECOND detector pass, merged with the episode pass.

    For each annotated segment whose label implies contact (CONTACT_LABEL), the
    aperture detector re-runs on the padded window with WINDOW-relative
    thresholds. A window event is accepted only if it (a) falls inside the
    labeled segment, (b) is not a duplicate of an episode-pass event, and
    (c) keeps the hand's close/open sequence alternating (episode-pass events
    win collisions; a boosted leading open with no prior close is dropped).

    Returns (merged_events, boosted) — boosted lists the accepted additions for
    diagnostics/trace provenance."""
    T = len(thumb_pos)
    pad = int(fps) if pad is None else pad
    candidates = []
    for (s, e, lab) in segments:
        if not lab or not CONTACT_LABEL.search(str(lab)):
            continue
        lo, hi = max(0, int(s) - pad), min(T, int(e) + pad)
        if hi - lo < 2 * min_dwell:
            continue
        ev, _ = aperture_events(thumb_pos[lo:hi], index_pos[lo:hi], middle_pos[lo:hi],
                                fps=fps, min_dwell=min_dwell, min_range=min_range,
                                drop_leading_open=False)
        for f, k in ev:
            g = f + lo
            if int(s) <= g < int(e):
                candidates.append((g, k))
    merged = list(base_events)
    accepted = []
    for g, k in sorted(set(candidates)):
        if any(k == k0 and abs(g - f0) <= 2 * min_dwell for f0, k0 in merged):
            continue                       # duplicate of an existing event
        merged.append((g, k))
        accepted.append((g, k))
    merged.sort()
    boosted_set = set(accepted)
    kept: list = []
    for f, k in merged:                    # enforce alternation; base events win
        if not kept:
            if k == "open" and (f, k) in boosted_set:
                continue                   # boosted leading open = reach uncurl
            kept.append((f, k))
            continue
        if k != kept[-1][1]:
            kept.append((f, k))
        elif (f, k) in boosted_set:
            continue                       # boosted collides with kept -> drop it
        elif kept[-1] in boosted_set:
            kept[-1] = (f, k)              # base wins over a boosted collision
    boosted = [e for e in kept if e in boosted_set]
    return kept, boosted
