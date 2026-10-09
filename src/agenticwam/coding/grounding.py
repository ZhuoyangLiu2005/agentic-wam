"""Gripper-event detection (sensor side) for the ASPIRE code-GT pipeline — vendored into AgenticWAM.

grip_transitions: ordered open<->closed gripper events for one arm, from the SIGN of the smoothed
gripper velocity (not absolute cutoffs) — the observed scaffold for the actor and the ground truth for
gripper-effect reconciliation. Self-contained (numpy only)."""
from __future__ import annotations
import numpy as np
import os


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")

GRIP_SMOOTH = 5
GRIP_V_EPS = 0.01
GRIP_MIN_RUN = 4
GRIP_MIN_RANGE = 0.08   # M1: below this the gripper is essentially static -> no transitions
GRIP_NORM_BELOW = 0.6   # M1: only normalize arms whose dynamic range is compressed below this
# M2: minimum DWELL (frames) a gripper must remain in a state before flipping back.
# A close/open round-trip shorter than this is a sensor TWITCH, not a real actuation
# (the raw velocity-sign detector otherwise emits a full close+open cycle for each
# blip -> per-frame open/close oscillation that dominates the code GT). Env-overridable;
# set 0 to disable (byte-identical to the pre-debounce detector).
GRIP_MIN_DWELL = int(os.environ.get("GRIP_MIN_DWELL", "20"))


def _debounce_events(ev, T, min_dwell=None):
    """Suppress spurious short-lived gripper flips. If the state entered by an event
    reverts within < min_dwell frames, that (close,open) / (open,close) pair is a
    twitch -> drop both. Iterated to convergence. Preserves the ordered, strictly
    alternating (frame, 'close'|'open') contract. min_dwell<=0 -> no-op."""
    md = GRIP_MIN_DWELL if min_dwell is None else min_dwell
    if md <= 0 or len(ev) < 2:
        return ev
    ev = list(ev)
    changed = True
    while changed and len(ev) >= 2:
        changed = False
        for k in range(len(ev)):
            end = ev[k + 1][0] if k + 1 < len(ev) else T
            if end - ev[k][0] < md:
                if k + 1 < len(ev):
                    del ev[k:k + 2]   # twitch: entered at k, reverts at k+1
                else:
                    del ev[k]         # brief flip at the very end
                changed = True
                break
    return ev


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
    # M1: per-episode ROBUST normalization to [0,1] (2/98 pct) so the velocity/run thresholds are
    # scale-invariant -- arms with a COMPRESSED dynamic range (some swing only ~0.18) keep their
    # transitions instead of losing them to the fixed deadband. A static signal yields nothing.
    lo, hi = np.percentile(g, 2.0), np.percentile(g, 98.0)
    rng = hi - lo
    if rng < GRIP_MIN_RANGE:
        return []
    # normalize ONLY compressed-range arms (rng < GRIP_NORM_BELOW) so slow transitions clear the fixed
    # deadband; well-behaved near-full-swing arms are left byte-identical to avoid churning correct
    # detections (data-relative velocity sign already handles them).
    gn = (g - lo) / rng if rng < GRIP_NORM_BELOW else g
    gs = smooth(gn, GRIP_SMOOTH)
    v = np.zeros(T)
    v[1:] = np.diff(gs)
    reg = np.where(v < -GRIP_V_EPS, -1, np.where(v > GRIP_V_EPS, 1, 0))  # -1 close, +1 open, 0 flat
    state = "open" if gs[0] > (gs.min() + gs.max()) / 2.0 else "closed"   # data-relative (unchanged)
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
    return _debounce_events(ev, T)
