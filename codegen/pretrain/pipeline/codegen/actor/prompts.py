"""Actor-prompt MECHANISM, shared across embodiments: the event-list renderer,
program context, and the coder message builder. The embodiment-specific prompt
TEXT (CODER_SYS / CONV / grip_and_motion_directive) lives in per-embodiment
modules (``prompts_yam.py`` / ``prompts_ego.py``, one file per embodiment —
ASPIRE per-suite prompt-tree convention), selected here at the import seam.

PROMPT-SENSITIVE — the episode cache key hashes CODER_SYS, CONV, API_DOC,
SKILLS_DOC AND `inspect.getsource` of `build_coder`, so any edit to those
(even a comment inside build_coder) correctly busts the per-run caches.

ONE VLM call per slice: the model sees the prompt AND the frames and writes
the code directly; there is no separate perceive/describe stage.
"""
from __future__ import annotations

import textwrap

from codegen.api.active import API_DOC, EMBODIMENT
from codegen.library import SKILLS_DOC
from .media import _img, _txt

if EMBODIMENT in ("yam", "yam_sft"):        # the prompt-module selection seam
    from . import prompts_yam as _emb
else:
    from . import prompts_ego as _emb

CODER_SYS = _emb.CODER_SYS
CODER_SYS_VANILLA = _emb.CODER_SYS_VANILLA
CONV = _emb.CONV


def _anchor_str(grips):
    """Ordered, authoritative gripper-EVENT sequence: every event numbered and
    timestamped (matching the EVENT captions bound to the frames), with a repeated
    same-arm 2-event cycle run-length compressed to one
    '(CLOSE, OPEN on LEFT) x12 (events 3-26, ...)' group — which is exactly what
    one `for _ in range(12)` close/rotate/open loop produces."""
    if not grips:
        return "(none — no gripper actuation was measured in this segment)"
    side = {"left": "LEFT", "right": "RIGHT"}
    ev = [(a, k) for (_f, a, k) in grips]
    fr = [f for (f, _a, _k) in grips]
    toks, i = [], 0
    while i < len(ev):
        # greedy: longest repeat of a same-arm (e1, e2) cycle starting here
        if (i + 3 < len(ev) and ev[i][0] == ev[i + 1][0] and ev[i][1] != ev[i + 1][1]
                and ev[i:i + 2] == ev[i + 2:i + 4]):
            unit, n = ev[i:i + 2], 2
            while ev[i + 2 * n:i + 2 * n + 2] == unit:
                n += 1
            toks.append(f"({unit[0][1].upper()}, {unit[1][1].upper()} on {side[unit[0][0]]}) x{n} "
                        f"(events {i + 1}-{i + 2 * n}, t{fr[i]}-t{fr[i + 2 * n - 1]})")
            i += 2 * n
        else:
            a, k = ev[i]
            toks.append(f"{i + 1}. {k.upper()} on {side[a]} @t{fr[i]}")
            i += 1
    return textwrap.fill(" -> ".join(toks), 96, initial_indent="  ", subsequent_indent="  ")


def _prog_context(prog_so_far):
    """DATA: the code from earlier subtasks to continue. The 'continue / new-lines-only'
    RULE lives in the system prompt, so this is just the labeled program text."""
    if not prog_so_far.strip():
        return "PROGRAM SO FAR: (none — this is the first subtask.)\n\n"
    return f"PROGRAM SO FAR (continue it; emit only the new lines):\n{prog_so_far}\n\n"


def _grip_and_motion_directive(grips):
    """Event DATA + motion guidance in the ACTIVE embodiment's vocabulary
    (delegates to the per-embodiment prompt module; the anchor rendering is
    shared). NOTE this text is NOT part of the cache key — an embodiment's
    module must keep its output byte-stable unless silently-stale caches are
    intended."""
    return _emb.grip_and_motion_directive(grips, _anchor_str(grips))


def build_coder(frame_items, label, grips, instruction, prog_so_far="",
                vanilla=False):
    """The ONE coder prompt: the VLM sees the frames AND writes the slice's code.

    Layout: every STATIC block (rules + API + frozen skills + conventions) lives
    in the SYSTEM message — one byte-identical prefix across all slices and
    episodes of a pipeline version. The USER message is the slice's narrative,
    in reading order: task -> this subtask -> the job -> the frames (with
    detector events CAPTIONED onto the frame just before each) ->
    the program to continue -> the authoritative event contract LAST.

    ``vanilla`` = annotation-only mode: NO detector content anywhere — the
    vanilla system prompt, no event captions, and the vanilla directive instead
    of the event list. Callers pass grips=[] so no event pre-frames exist."""
    skills_part = f"\n{SKILLS_DOC}\n" if SKILLS_DOC else ""
    sys_text = CODER_SYS_VANILLA if vanilla else CODER_SYS
    system = f"{sys_text}\n\nAPI (call ONLY these):\n{API_DOC}\n{skills_part}\n{CONV}"
    unit = "hand" if EMBODIMENT == "ego" else "gripper"
    head = (f"This subtask (human annotation): '{label}'"
            if label else "This is an UNLABELED transition segment")
    frames_line = ("The frames, sampled along the slice:" if vanilla else
                   f"The frames. Detector events are captioned on the frame taken just "
                   f"before the {unit} actuates:")
    pre = (f"Overall task: {instruction}\n"
           f"{head} — frames below.\n\n"
           "Your job: read the frames and CONTINUE the program with this "
           "subtask's code — output only the new lines.\n\n"
           f"{frames_line}")
    # bind each observed event to its pre-frame via the frame's generated name
    n_ev = len(grips)
    caption = {} if vanilla else {
        f"pre-{a}-{k}@{f}": (f"EVENT {i} of {n_ev} — the detector reports "
                             f"{k.upper()} on {a.upper()} at t={f}.")
        for i, (f, a, k) in enumerate(grips, 1)
    }
    ask = (f"{_prog_context(prog_so_far)}"
           f"{_emb.vanilla_directive() if vanilla else _grip_and_motion_directive(grips)}")
    user = [_txt(pre)]
    ph = []
    for (name, fi, b64, rel) in frame_items:
        cap = caption.get(name)
        lab = (f"{cap}\nThis frame (t={fi}) is the moment just before it:"
               if cap else f"{name} frame (t={fi}):")
        user += [_txt(lab), _img(b64)]
        ph.append(f"\n[{cap or name} frame t={fi} -> {rel}]")
    user.append(_txt(ask))
    text = pre + "".join(ph) + "\n\n" + ask
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], text
