#!/usr/bin/env python
"""Gripper-event detection for the YAM inverse code-gen pipeline.

Import-only library (no CLI). Exposes ``grip_transitions`` — the ordered open<->closed
gripper events for one arm, derived from the SIGN of the smoothed gripper velocity. 
Used by ``codegen.pipeline.episode`` as the observed gripper scaffold handed to the coding agent.
"""
from __future__ import annotations


from pathlib import Path

import numpy as np


from .segment import smooth

# Velocity-based gripper event detection:
# events come from the SIGN of the smoothed gripper velocity, not absolute cutoffs.
GRIP_SMOOTH = 5        # frames, boxcar smoothing of the analog gripper signal
GRIP_V_EPS = 0.01      # |d(grip)/frame| below this = flat (velocity dead-band)
GRIP_MIN_RUN = 4       # a closing/opening ramp must persist >= this many frames to count

# --- hover-chatter suppression (post-detection; see debounce_events) ----------
#: consecutive events closer than this many frames belong to the same BURST
GRIP_MIN_DWELL = 20
#: a burst counts as HOVER only if the TCP moved less than this (metres) across it
GRIP_HOVER_EPS = 0.02


def grip_transitions(grip):
    """ALL gripper open<->closed transitions for one arm, from the SIGN of the (smoothed)
    gripper VELOCITY rather than absolute thresholds. A sustained fall (velocity below
    -GRIP_V_EPS for >= GRIP_MIN_RUN frames) is a CLOSE, pinned to the knee where the ramp
    flattens (~contact); a sustained rise is an OPEN, pinned to where the plateau departs
    (~release). Returns the ordered list of (frame, "close"|"open")."""
    g = np.asarray(grip, dtype=np.float64)
    T = len(g)
    if T < 2:
        return []
    gs = smooth(g, GRIP_SMOOTH)
    v = np.zeros(T)
    v[1:] = np.diff(gs)
    reg = np.where(v < -GRIP_V_EPS, -1, np.where(v > GRIP_V_EPS, 1, 0))  # -1 close, +1 open, 0 flat
    # initial open/closed from the signal's own range (data-relative, not a fixed cutoff)
    state = "open" if gs[0] > (gs.min() + gs.max()) / 2.0 else "closed"
    ev = []
    i = 1
    while i < T:
        j = i
        while j < T and reg[j] == reg[i]:
            j += 1
        if reg[i] == -1 and (j - i) >= GRIP_MIN_RUN and state != "closed":
            ev.append((j - 1, "close"))   # knee: end of the falling ramp ~ contact
            state = "closed"
        elif reg[i] == 1 and (j - i) >= GRIP_MIN_RUN and state != "open":
            ev.append((i, "open"))        # start of the rising ramp ~ release
            state = "open"
        i = j
    return ev


def _bursts(ev, min_dwell: int):
    """Maximal runs of events whose consecutive gaps are all < ``min_dwell``."""
    out, cur = [], [0]
    for i in range(1, len(ev)):
        if ev[i][0] - ev[i - 1][0] < min_dwell:
            cur.append(i)
        else:
            out.append(cur)
            cur = [i]
    out.append(cur)
    return [b for b in out if len(b) > 1]


def debounce_events(ev, pos=None, min_dwell: int | None = None,
                    hover_eps: float | None = None):
    """Collapse HOVER CHATTER: a rapid open/close/open/close burst produced while
    the arm sits in ONE place, which the velocity detector cannot tell from a real
    grasp-release.

    Two conditions must BOTH hold before anything is removed:

      * **rapid** — every gap inside the burst is shorter than ``min_dwell`` frames
        (20 = 0.67 s at 30 fps), and
      * **stationary** — the TCP moved less than ``hover_eps`` metres across the
        whole burst. This is the part that makes it safe: a genuine fast re-grasp
        travels (approach, lift, reposition), hovering does not. Without a position
        test a pure dwell rule also deletes real quick re-grasps.

    A burst is then reduced to its NET state change, which is what "delete the ones
    in the middle" means once you look at it as a state machine: an even-length
    burst returns the gripper to where it started, so all of it goes; an odd-length
    burst is one real transition wrapped in chatter, so the FIRST event survives
    (standard debounce convention — the leading edge is when it actually happened).
    Either way the ordered, strictly-alternating event contract is preserved.

    Set ``GRIP_MIN_DWELL=0`` to disable: the output is then byte-identical to the
    raw detector, which is what makes an A/B possible.

    Returns ``(events, dropped)`` — ``dropped`` is every removed event, kept for the
    QC view and for auditing what the filter took away.
    """
    from codegen import config
    if min_dwell is None:
        min_dwell = int(config.get("GRIP_MIN_DWELL") or 0)
    if hover_eps is None:
        hover_eps = float(config.get("GRIP_HOVER_EPS") or 0.0)
    ev = list(ev)
    if min_dwell <= 0 or len(ev) < 2:
        return ev, []
    p = None if pos is None else np.asarray(pos, dtype=np.float64)
    drop = set()
    for burst in _bursts(ev, min_dwell):
        if p is not None:
            f0, f1 = ev[burst[0]][0], ev[burst[-1]][0]
            seg = p[max(0, min(f0, len(p) - 1)):min(f1 + 1, len(p))]
            if len(seg) and float(np.linalg.norm(seg.max(0) - seg.min(0))) >= hover_eps:
                continue                     # the arm travelled — a real manoeuvre
        keep_first = len(burst) % 2 == 1     # odd burst = one net transition
        drop.update(burst[1:] if keep_first else burst)
    return ([e for i, e in enumerate(ev) if i not in drop],
            [e for i, e in enumerate(ev) if i in drop])
