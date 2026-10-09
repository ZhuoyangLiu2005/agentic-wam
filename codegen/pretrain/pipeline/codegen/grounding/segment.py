#!/usr/bin/env python
"""Deterministic slicing/segmentation helpers for the YAM inverse code-gen pipeline.

Import-only library (no CLI). These primitives turn a converted episode's per-arm
proprioception + per-frame ``annotation.sub_task`` labels into the structure the
VLM pipeline (``codegen.pipeline.episode``) fills in:

  * ``sub_task_blocks`` — contiguous runs of the per-frame sub_task id -> ordered
    (start, end, label) semantic blocks, with annotation noise filtered. This is
    what actually SLICES the trajectory into sub-tasks.
  * ``runs`` / ``smooth`` — run-length encoding + boxcar smoothing helpers (``smooth``
    is also used by ``grip_transitions`` in transitions.py).
"""
from __future__ import annotations

import numpy as np

# --- tunables (frames @ 30 Hz) ------------------------------------------------
MIN_BLOCK = 8              # drop sub_task runs shorter than this (annotation noise)

#: Deliberately-discarded annotation labels — matched as EXACT labels
#: (normalized: stripped, lowercased), never as substrings: the old substring
#: match nulled every real label containing "trim" ("Pick up a hair trimmer."
#: lost 5 of 6 labels in one episode). The per-embodiment spec may override
#: this set (sub_task_blocks(..., noise_labels=spec.noise_labels)).
NOISE_LABELS = frozenset({"mistake", "trim", "reset both arms",
                          "unsure_how_to_annotate"})


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def runs(lab: np.ndarray, lo: int, hi: int):
    """Yield (start, end_exclusive, label) over the maximal runs within [lo, hi)."""
    i = lo
    while i < hi:
        j = i
        while j < hi and lab[j] == lab[i]:
            j += 1
        yield i, j, int(lab[i])
        i = j


def unmapped_ids(sub_id: np.ndarray, text: dict) -> list[int]:
    """Sub-task ids PRESENT in the per-frame data but ABSENT from the id->text
    map — the signature of a broken/missing mapping (e.g. reading a selection
    layer without ``meta/``), as opposed to a genuinely unannotated episode.

    Id 0 is exempt by convention: the yam map carries 0 -> "" (so it is never
    unmapped there), and the readers use 0 as the explicit UNLABELED
    sentinel. A genuinely unannotated episode therefore reports [] here, while
    a broken mapping reports the ids the data claims but the map cannot name."""
    return sorted({int(i) for i in np.unique(sub_id)} - {int(k) for k in text} - {0})


def sub_task_blocks(sub_id: np.ndarray, text: dict, noise_labels=NOISE_LABELS):
    """Contiguous runs of the per-frame sub_task id -> ordered
    (start, end, label, noise_label).

    ``label`` is None for anything that should not head a code block; when the
    REASON is a noise annotation (EXACT match on the normalized label — e.g.
    "reset both arms"), the original text is preserved in ``noise_label`` so
    the pipeline can distinguish "the annotator SAID this span is noise"
    (trust it: skip) from "nobody said anything" (a gap: judge by events).
    Runs shorter than MIN_BLOCK and empty labels are plain gaps."""
    blocks = []
    for s, e, tid in runs(sub_id, 0, len(sub_id)):
        label = text.get(int(tid), "").strip()
        low = label.lower()
        if low in noise_labels:
            blocks.append((s, e, None, label))          # annotated noise
        elif (not label) or (e - s) < MIN_BLOCK:
            blocks.append((s, e, None, None))           # gap / too short
        else:
            blocks.append((s, e, label, None))
    return blocks
