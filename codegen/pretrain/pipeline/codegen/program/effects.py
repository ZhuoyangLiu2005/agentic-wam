"""Gripper transitions IMPLIED by code + their match against the observed
sensor sequence — the grounder's core (consumes the API's gripper-effect
declarations)."""
from __future__ import annotations

from codegen.api.active import resolve_effects
from .calls import effective_api_calls


def implied_grip_events(code):
    """Gripper transitions IMPLIED by the code's calls -- primitive (`open/close_gripper`),
    semantic (`pick`/`place`/...), or a composed SKILL call (expanded through its body)
    -- via the API's gripper-event CONTRACTS. This is what makes grounding work when
    the open/close is HIDDEN inside a semantic or composed call.

    Returns ordered ``[(offset, arm, kind)]`` where offset is the 0-based index into `code`
    of the producing call (a skill's events all pin to its call-site line), or ``None``
    if the code doesn't parse."""
    calls = effective_api_calls(code)
    if calls is None:
        return None
    events = []
    for (off, _col, _seq, name, bound) in calls:
        for arm, kind in (resolve_effects(name, bound) or []):
            if arm in ("left", "right"):     # unresolved arm (non-literal) -> skip -> mismatch
                events.append((off, arm, kind))
    return events


def match_grippers(code, grips):
    """Align the OBSERVED gripper transitions (sensor) to the transitions IMPLIED by the
    code's calls -- via the API's gripper-event contracts, so a semantic ``pick``/``place``
    grounds as well as an explicit ``open_gripper``/``close_gripper``. The code is NEVER rewritten:

      * MATCHED (implied == observed: same kinds, arms, ORDER): return per-event anchors
        ``[(code_offset, true_frame)]`` pinning each transition frame to the LINE of the
        call that produces it (a `pick` line, a `close_gripper` line, ...).
      * MISMATCH / unparseable: NO anchors -> frame map falls back to block-level.

    ``grips`` = sorted [(frame, arm, kind)]. Returns (anchors, report)."""
    implied = implied_grip_events(code)
    observed = [(k, a) for (_f, a, k) in grips]
    coder = None if implied is None else [(kind, arm) for (_off, arm, kind) in implied]
    report = {"coder_grippers": [f"{k}_{a}" for (k, a) in (coder or [])],
              "observed_grippers": [f"{k}_{a}" for (k, a) in observed],
              "observed_frames": [int(f) for (f, _a, _k) in grips],
              "matched": coder == observed}
    if report["matched"]:
        return [(implied[i][0], int(grips[i][0])) for i in range(len(grips))], report
    return [], report


#: the explicit open/close primitives of the ACTIVE embodiment, resolved from
#: the API rather than hardcoded (yam: *_gripper, ego: *_hand)
def _primitive(kind: str) -> str:
    from codegen.api.active import EMBODIMENT, _API_SIG
    name = f"{kind}_gripper" if EMBODIMENT in ("yam", "yam_sft") else f"{kind}_hand"
    if name not in _API_SIG:
        raise KeyError(f"{name} is not in the {EMBODIMENT} API")
    return name


def _prims_for(tokens):
    """['close_left', ...] -> the code lines that produce exactly those events."""
    out = []
    for tok in tokens:
        kind, _, arm = tok.partition("_")
        if kind not in ("open", "close") or arm not in ("left", "right"):
            return None
        out.append(f"robot.{_primitive(kind)}('{arm}')")
    return out


REPAIR_NOTE = ("# [auto-repair] gripper events under-emitted by the coder; "
               "inserted from the sensor (grounded):")


def repair_tail(code, grips):
    """Deterministically repair UNDER-EMISSION at the ends of a slice.

    The coder's most common near-miss is dropping the first or last gripper event
    of a slice — everything it wrote is right, there is simply less of it. When
    the implied sequence is a CONTIGUOUS SUBLIST of the observed one, the missing
    head/tail events are unambiguous, so they are inserted as explicit primitives
    (taken from the sensor, marked as such) and the slice is re-matched. Ported
    from the collaborator's pipeline, where it moved reconciliation from 48.6% to
    71.0% over a real 1000-episode run.

    Two things this deliberately does NOT do: it never touches the MIDDLE of a
    slice (a gap there is a real disagreement, not a truncation), and it reverts
    entirely unless the repair actually makes the slice match.

    The inserted lines are NOT the model's work. They are marked with
    ``REPAIR_NOTE`` in the program and counted per slice as ``tail_repaired`` so
    the reconciliation rate can always be reported both ways.

    Returns ``(code, n_inserted)``."""
    if effective_api_calls(code) is None:
        return code, 0
    _anchors, rep = match_grippers(code, grips)
    if rep["matched"]:
        return code, 0
    coder, observed = rep["coder_grippers"], rep["observed_grippers"]
    n = len(coder)
    if n >= len(observed):
        return code, 0                      # over-emission, not under-emission
    pos = next((q for q in range(0, len(observed) - n + 1)
                if observed[q:q + n] == coder), -1)
    if pos < 0:
        return code, 0                      # not a contiguous sublist -> ambiguous
    pre, post = _prims_for(observed[:pos]), _prims_for(observed[pos + n:])
    if pre is None or post is None:
        return code, 0
    repaired = (([REPAIR_NOTE] + pre) if pre else []) + list(code) \
        + (([REPAIR_NOTE] + post) if post else [])
    _a2, rep2 = match_grippers(repaired, grips)
    return (repaired, len(pre) + len(post)) if rep2["matched"] else (code, 0)


#: the EXPLICIT open/close primitives — the only lines the over-emission lint may
#: touch. A semantic verb (`pick`, `place`, `handover`) carries plan meaning and is
#: never removed, even when its transitions are the redundant ones.
def _explicit_primitives() -> set:
    return {_primitive("open"), _primitive("close")}


LINT_NOTE = ("# [over-emission lint] removed a redundant explicit gripper call "
             "(this episode did not have that transition):")


def lint_over_emission(code, grips, max_removals: int = 2):
    """Deterministically repair OVER-emission: the coder implied MORE transitions
    than this episode has.

    The mirror of ``repair_tail``, and on reused skeletons it matters just as
    much: measured on 6,426 real blocks assembled from a per-label cache, 29.2%
    were short of events (repair_tail's direction) and **24.9% had too many**,
    with another 8.5% where the block has no events at all. Without this pass
    that third of the corpus can never earn a pin.

    Search, not surgery: comment out one explicit gripper line, then two, and keep
    the first removal that makes the slice reconcile. Only ever an explicit
    ``open/close`` primitive — a semantic call is plan content and stays. Lines
    are COMMENTED, not deleted, so line numbering (and therefore every existing
    frame->line pin) is preserved.

    Returns ``(code, n_lines_commented)``; unchanged if nothing helps."""
    import itertools
    calls = effective_api_calls(code)
    if calls is None or match_grippers(code, grips)[1]["matched"]:
        return code, 0
    prim = _explicit_primitives()
    name_of_line = {}
    for (off, _col, _seq, name, _b) in calls:
        name_of_line.setdefault(off, name)
    expl = sorted(off for off, nm in name_of_line.items() if nm in prim)
    if not expl:
        return code, 0

    def commented(drop):
        return [(f"{LINT_NOTE} {ln.strip()}") if i in drop else ln
                for i, ln in enumerate(code)]

    for r in range(1, max(1, max_removals) + 1):
        for sub in itertools.combinations(expl, r):
            cand = commented(set(sub))
            if match_grippers(cand, grips)[1]["matched"]:
                return cand, len(sub)
    return code, 0
