"""Frame sampling + encoding for VLM calls (message-content primitives)."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def _to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(frame, dtype=np.uint8))


def frame_b64(img: Image.Image) -> str:
    # JPEG q90, full resolution: ~5x smaller than PNG so slices with many gripper events
    # (90 events -> 91 frames -> ~35 MB PNG) stay under the Azure/Claude 32 MB request cap.
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _txt(t):
    return {"type": "text", "text": t}


def _img(b64):
    return {"type": "image_url", "image_url": {"url": b64}}


def gap_fill(points, max_gap):
    """Given sorted anchor frame indices, insert evenly spaced frames into every
    over-gap interval so no two consecutive kept frames are more than ``max_gap``
    apart (an interval of length L gets ceil(L/max_gap)-1 inserted points)."""
    out = []
    for a, b in zip(points, points[1:]):
        out.append(int(a))
        gap = b - a
        if gap > max_gap:
            k = int(np.ceil(gap / max_gap))          # subintervals, each <= max_gap
            out.extend(int(round(a + gap * j / k)) for j in range(1, k))
    out.append(int(points[-1]))
    return out


def _even(seq, k):
    """``k`` items spread evenly over ``seq``, first and last always kept."""
    if k <= 0:
        return []
    if len(seq) <= k:
        return list(seq)
    step = (len(seq) - 1) / (k - 1) if k > 1 else 0
    return [seq[i] for i in sorted({int(round(i * step)) for i in range(k)})]


def _cap(items, anchors, k):
    """Trim a sampled slice to at most ``k`` frames.

    GAP FILLERS go first (thinned evenly), because they are interchangeable
    padding; ANCHORS (slice endpoints + the pre-frame of each observed event)
    survive until the anchors ALONE exceed the cap, and even then the endpoints
    are kept. Without a cap a single long unlabeled span produces a request that
    exceeds the router's 32 MB body limit (measured: a 5,197-frame span at
    gap=40 samples 218 frames ~= 16.6 MB of JPEG before base64), and the whole
    episode dies — one failed episode being, today, an episode that silently
    leaves the frontier for good."""
    anchor_items = [it for it in items if it[0] in anchors]
    fillers = [it for it in items if it[0] not in anchors]
    if len(anchor_items) >= k:
        keep = _even(anchor_items, k)
    else:
        keep = anchor_items + _even(fillers, k - len(anchor_items))
    return sorted(keep)


def subtask_frames(bs, be, max_gap, grips, T, pad=4, label_events=True,
                   max_frames=0):
    """Frames for a sub-task VLM call, spaced by MAX GAP (not a fixed count —
    slices differ in length). Anchors are the slice endpoints PLUS, for each
    observed gripper transition, the single PRE frame (``pad`` frames BEFORE the
    event — the moment just before the gripper actuates, before any post-grasp
    occlusion). Every interval between consecutive anchors longer than
    ``max_gap`` is filled with evenly spaced frames until all gaps fit (with no
    events this degenerates to uniform sampling). Returns an ordered,
    de-duplicated [(name, idx)]; endpoint names win over gripper-frame names.

    ``label_events=False``: event anchors still steer WHERE frames are taken
    but are named neutrally (``t<idx>``, indistinguishable from gap fillers) —
    for annotation-only prompts, where nothing may reveal detector output.

    ``max_frames`` (0 = unlimited) is the HARD per-request ceiling; see ``_cap``
    for what gets dropped first."""
    end = max(be - 1, bs)
    named = [(bs, "START")] + ([] if end == bs else [(end, "END")])
    for (f, a, k) in grips:
        gi = int(np.clip(f - pad, bs, end))
        named.append((gi, f"pre-{a}-{k}@{f}" if label_events else f"t{gi}"))
    seen = {}
    for idx, name in named:              # first occurrence wins (endpoints added first)
        seen.setdefault(int(np.clip(idx, 0, T - 1)), name)
    anchors = set(seen)                  # everything below this line is padding
    max_gap = max(int(max_gap), 1)
    for idx in gap_fill(sorted(seen), max_gap):
        seen.setdefault(idx, f"t{idx}")  # gap fillers; anchors keep their names
    items = sorted(seen.items())
    if max_frames and len(items) > int(max_frames):
        items = _cap(items, anchors, int(max_frames))
    return [(name, idx) for idx, name in items]
