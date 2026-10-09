"""Label inventory — which labels exist, and which episode should teach each one.

Stage A generates one skeleton per label from ONE episode's frames. WHICH episode
is not a detail: that episode's execution becomes the canonical form for every
other episode carrying the same label. Taking whichever episode a shard happens
to reach first means an atypical take (a fumble, an unusual
object placement) can become the template for a hundred others.

Here the representative is chosen: among a label's occurrences, the one whose
gripper-event count is the MEDIAN, breaking ties on duration closest to median.
Same cost, and the canonical form is the typical one.

Building the inventory reads two parquet columns per episode and no video.
"""
from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

import numpy as np


def _episode_blocks(spec, task: str, episode_index: int):
    """(label, bs, be, n_events, duration) for one episode — parquet only.

    Goes through ``spec.load_state`` for the same reason Stage B does: the
    layout and the detector are per-embodiment, and the two tiers MUST see the
    same events — the signature recorded here is what Stage B is later scored
    against."""
    from codegen.grounding.facts import block_events
    from codegen.grounding.segment import sub_task_blocks

    st = spec.load_state(task, episode_index)
    out = []
    for (bs, be, label, noise) in sub_task_blocks(st["sub_id"], st["text"],
                                                 noise_labels=spec.noise_labels):
        if not label:
            continue
        sig = tuple(f"{k}_{a}" for (_f, a, k) in block_events(st["per_arm"], bs, be))
        out.append({"label": label, "bs": int(bs), "be": int(be),
                    "n_events": len(sig), "sig": sig, "duration": int(be - bs)})
    return out


def build(units, embodiment: str = "yam", log=print, limit_episodes: int = 0):
    """Scan work units -> {label: {occurrences, representative, signatures}}.

    ``units`` is the (task, episode_index) list the campaign will annotate; the
    inventory only has to cover the labels those units actually contain."""
    from codegen.embodiments import get_spec
    spec = get_spec(embodiment)
    if limit_episodes:
        units = units[:limit_episodes]
    labels: dict[str, dict] = {}
    seen = skipped = 0
    errors: collections.Counter = collections.Counter()
    first_error = None
    for n, (task, ei) in enumerate(units, 1):
        try:
            blocks = _episode_blocks(spec, task, ei)
        except Exception as e:                       # a unit we cannot read is not
            skipped += 1                             # a reason to lose the scan
            errors[f"{type(e).__name__}: {e}"[:200]] += 1
            if first_error is None:
                first_error = f"{task}:{ei}: {type(e).__name__}: {e}"
            if skipped <= 3:
                log(f"[inventory] skip {task}:{ei}: {type(e).__name__}: {e}")
            continue
        seen += 1
        for b in blocks:
            rec = labels.setdefault(b["label"], {"occurrences": [], "sigs": collections.Counter()})
            rec["occurrences"].append({"task": task, "episode": int(ei),
                                       "bs": b["bs"], "be": b["be"],
                                       "n_events": b["n_events"],
                                       "duration": b["duration"]})
            rec["sigs"][b["sig"]] += 1
        if n % 25 == 0 or n == len(units):
            log(f"[inventory] {n}/{len(units)} episodes, {len(labels)} labels")

    # A scan that read NOTHING is a configuration/embodiment mismatch, never a
    # data gap — and it used to look like success: every unit's exception was
    # swallowed, `{}` was written, and the CLI exited 0. Fail loud instead.
    if units and seen == 0:
        for msg, k in errors.most_common(6):
            log(f"[inventory]   {k:6d}x  {msg}")
        raise RuntimeError(
            f"all {len(units)} unit(s) unreadable — first: {first_error}. "
            f"Nothing was scanned, so an empty inventory here would be a lie.")
    if errors:                       # a PARTIAL loss is survivable but never quiet
        log(f"[inventory] {skipped}/{len(units)} unit(s) unreadable, by error:")
        for msg, k in errors.most_common(6):
            log(f"[inventory]   {k:6d}x  {msg}")

    for label, rec in labels.items():
        occ = rec["occurrences"]
        med_ev = statistics.median(o["n_events"] for o in occ)
        med_du = statistics.median(o["duration"] for o in occ)
        # the TYPICAL take: median event count first (it decides the contract the
        # skeleton will imply), duration only as a tie-break
        rec["representative"] = min(
            occ, key=lambda o: (abs(o["n_events"] - med_ev), abs(o["duration"] - med_du)))
        rec["n"] = len(occ)
        # the modal event signature, and how much of the label agrees with it —
        # this is the ceiling on how often the skeleton can pin without repair
        sig, k = rec["sigs"].most_common(1)[0]
        rec["modal_sig"] = list(sig)
        rec["modal_share"] = round(k / len(occ), 3)
        del rec["sigs"]
    log(f"[inventory] {seen} episodes read ({skipped} unreadable), "
        f"{len(labels)} distinct labels, "
        f"{sum(r['n'] for r in labels.values())} block instances")
    return labels


def save(labels: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=1))


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())
