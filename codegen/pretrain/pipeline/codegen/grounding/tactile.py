"""Tactile contact-event detection for SHARPA teleop episodes.

Ported verbatim from the cap-x dex pipeline's event detector. Keep the two in sync: editing one
silently forks the grounding contract, so programs would be graded against different events.

Interface parity with grounding/aperture.py: `tactile_events(...)` returns
(events, dropped) where events is ordered [(frame, "close"|"open")] for ONE hand, so
segment/facts/effects/gate consume it unchanged. `dropped` is always [] -- tactile is a direct
sensor, so there is no splay/rest-curl curation to report.

Sharpa fingers carry 6-axis F/T; only the force axes [0:3] are used. Grasp = OPPOSITION:
thumb AND (index OR middle).
"""
from __future__ import annotations

import numpy as np

def _merge_min_dwell(segs: list[list], total_len: int, min_dwell: int) -> list[list]:
    """Remove interior [start, state] segments shorter than min_dwell, re-merging
    equal neighbors, so blips vanish instead of generating event pairs."""
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for j in range(1, len(segs)):
            end = segs[j + 1][0] if j + 1 < len(segs) else total_len
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
    return segs
def _mask_segments(mask: np.ndarray, min_dwell: int) -> list[list]:
    """[[start, bool_state], ...] runs of a boolean mask, dwell-filtered."""
    segs = [[0, bool(mask[0])]]
    for i in range(1, len(mask)):
        if bool(mask[i]) != segs[-1][1]:
            segs.append([i, bool(mask[i])])
    return _merge_min_dwell(segs, len(mask), min_dwell)
def _force_span(f6_series: np.ndarray) -> float:
    mag = np.linalg.norm(np.asarray(f6_series)[:, :3], axis=1)
    return float(np.percentile(mag, 99) - np.percentile(mag, 10))
def contact_mask(f6_series: np.ndarray, frac: float = 0.15,
                 span_floor: float = 0.0) -> np.ndarray:
    """Per-frame contact of ONE fingertip from its 6-axis F/T series (T, 6).

    Force axes only ([0:3]); threshold is baseline-relative (idle = 10th pct,
    span to the 99th pct). ``span_floor`` guards against an IDLE finger:
    pure sensor noise also has a "span", so the caller passes a floor derived
    from the episode's strongest finger; a finger whose span is below it never
    contacts."""
    mag = np.linalg.norm(np.asarray(f6_series)[:, :3], axis=1)
    idle = np.percentile(mag, 10)
    span = np.percentile(mag, 99) - idle
    if span < max(span_floor, 1e-3):
        return np.zeros(len(mag), bool)
    return (mag - idle) > frac * span
def tactile_events(f6_by_finger: dict, min_dwell: int = 10, span_floor: float = 0.0):
    """Ordered contact transitions of ONE hand from fingertip tactile (robot data).

    A grasp is OPPOSITION: the event series is thumb AND (index OR middle)
    co-contact — a lone finger resting on something is not a grasp (SKILL.md:
    hold vs. task contact). close = co-contact onset, open = offset, both dwell-
    filtered. Tactile is a direct sensor, so there is no splay/rest-curl curation;
    returns (events, dropped) with dropped always [] to mirror aperture_events."""
    thumb = contact_mask(f6_by_finger["thumb"], span_floor=span_floor)
    opp = (contact_mask(f6_by_finger["index"], span_floor=span_floor)
           | contact_mask(f6_by_finger["middle"], span_floor=span_floor))
    co = thumb & opp
    segs = _mask_segments(co, min_dwell)
    events = [(int(f), "close" if on else "open") for f, on in segs[1:]]
    return events, []
def tactile_hand_events(tactile: dict, min_dwell: int = 10, floor_frac: float = 0.05,
                        min_span_n: float = 1.0):
    """Per-hand tactile events: tactile = {"left": {finger: (T,6)}, "right": ...}.

    The idle-finger gate is two-layered, mirroring ``calibrated_hand``:
    relative — a finger participates only if its force span reaches
    ``floor_frac`` of the episode's strongest finger across BOTH hands (an idle
    hand's noise "span" is orders of magnitude below any real contact, but is
    not zero) — and absolute — the span must also rise ``min_span_n`` newtons
    above baseline (2026-07-18 sensor audit: real contacts span 1.7-16 N,
    idle/frozen noise < 0.35 N). The absolute floor is what makes a
    gesture-only episode safe: with no contact anywhere, the strongest finger
    span IS noise, so the relative floor is 5% of noise and every wiggle above
    it would otherwise become a phantom event."""
    ref = max(_force_span(f6) for fingers in tactile.values() for f6 in fingers.values())
    floor = max(floor_frac * ref, min_span_n)
    per_hand, dropped = {}, {}
    for hand, fingers in tactile.items():
        ev, dr = tactile_events(fingers, min_dwell=min_dwell, span_floor=floor)
        per_hand[hand], dropped[hand] = ev, dr
    return per_hand, dropped
TACTILE_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
