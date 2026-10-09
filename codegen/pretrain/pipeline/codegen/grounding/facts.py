"""Per-slice observed events (sensor side, numpy only)."""
from __future__ import annotations


def block_events(per_arm, bs, be):
    """The slice's ordered gripper/contact events from the detector, as
    (frame, arm, kind) tuples. The events are rendered ONCE, in the OBSERVED
    list (_anchor_str); nothing else from the sensor side reaches the prompt
    — the frames are the evidence."""
    grips = []
    for a in ("left", "right"):
        for (f, k) in per_arm[a]["grip_events"]:
            if bs <= f < be:
                grips.append((f, a, k))
    grips.sort()
    return grips
