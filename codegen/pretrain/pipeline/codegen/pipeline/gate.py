"""Reconciliation gate: score ONE finished run dir against the per-task
RECONCILIATION METRIC (program compiles AND every slice's gripper
reconciliation matched). Shared by the batch driver, the coordinator's audits,
and the (future) fix-agent's self-check — none of them recompute grounding,
they read what the episode run recorded (codegen.pipeline.artifacts schema).
"""
from __future__ import annotations

import json
from pathlib import Path

from codegen.pipeline import artifacts

ROOT = Path(__file__).resolve().parents[2]          # workspace root
DATA = ROOT / "outputs"


def run_dir_for(task: str, tag: str, embodiment: str = "yam",
                model: str = "gemini-3.1-pro") -> Path:
    """Run dir for one (task, tag) — layout knowledge lives in the embodiment
    spec (codegen.embodiments). Historical tiered YAM runs live under
    vlm_s2_subtask_fused_*: gate those by explicit path."""
    from codegen.embodiments import get_spec
    return get_spec(embodiment).run_dir(task, tag, model)


def gate_task(task: str, rd: Path) -> dict:
    """Score one finished run dir with the reconciliation metric -> manifest row."""
    row = {"task": task, "episode": rd.parent.name,
           "run_dir": str(rd.relative_to(ROOT))}
    prog, tr_p = rd / artifacts.PROGRAM, rd / artifacts.TRACE
    lines_p = artifacts.lines_path(rd)
    if not (prog.exists() and tr_p.exists() and lines_p.exists()):
        row.update(reconciled=False, error="missing run outputs")
        return row
    try:
        compile(prog.read_text(), str(prog), "exec")
        compiles = True
    except SyntaxError as e:
        compiles = False
        row["compile_error"] = f"line {e.lineno}: {e.msg}"
    tr = json.loads(tr_p.read_text())
    lj = json.loads(lines_p.read_text())
    all_slices = tr.get("slices", [])
    # skipped slices (annotated noise / event-free gaps — see pipeline/episode)
    # are EXCLUDED from reconciliation and accounted separately: their events
    # were never given to a VLM, so scoring them would be either a fake
    # mismatch or a fake pass.
    skipped = [s for s in all_slices if s.get("skipped")]
    slices = [s for s in all_slices if not s.get("skipped")]
    matched = [s["grip_reconcile"]["matched"] for s in slices]
    observed = sum(len(s["grip_reconcile"]["observed_grippers"]) for s in slices)
    mismatches = [
        {"slice": s["slice"], "label": s.get("label"),
         "coder": s["grip_reconcile"]["coder_grippers"],
         "observed": s["grip_reconcile"]["observed_grippers"]}
        for s in slices if not s["grip_reconcile"]["matched"]
    ]
    # Detector-pressure + pin-coverage counters. label_contact_zero_events is
    # the §4.1 flag (contact-verb label, zero observed events); computed here
    # from label+events so PRE-flag traces are covered too. pinned_slices =
    # slices that EARNED fine-grained frame->line pins (matched with >=1 event).
    from codegen.grounding.aperture import CONTACT_LABEL
    zero_ev = sum(1 for s in slices
                  if not s["grip_reconcile"]["observed_grippers"])
    label_zero = sum(1 for s in slices
                     if not s["grip_reconcile"]["observed_grippers"]
                     and s.get("label") and CONTACT_LABEL.search(s["label"]))
    pinned = sum(1 for s in slices
                 if s["grip_reconcile"]["matched"]
                 and s["grip_reconcile"]["observed_grippers"])

    # --- HONEST DENOMINATORS (added, never replacing the originals) -------------
    # `matched` is True for a slice with ZERO observed events: empty == empty. Those
    # slices carry no reconciliation signal at all, so a detector that goes quiet
    # RAISES the headline rate. The strict pair below is the same metric computed
    # only where there was something to match.
    with_ev = [s for s in slices if s["grip_reconcile"]["observed_grippers"]]
    matched_strict = sum(1 for s in with_ev if s["grip_reconcile"]["matched"])
    # `matched_raw` is the model's own score BEFORE deterministic tail repair
    # (pipeline/episode). Absent on pre-repair traces -> falls back to `matched`.
    matched_raw = sum(1 for s in slices
                      if s["grip_reconcile"].get("matched_raw",
                                                 s["grip_reconcile"]["matched"]))
    tail_ev = sum(s["grip_reconcile"].get("tail_repaired", 0) for s in slices)
    # ego only: how much of the "observed" stream came from the
    # annotation-conditioned second pass. Those events are NOT independent of the
    # annotation the actor was shown, so a base-only count is the conservative one.
    boosted = {(sd, int(f), str(k))
               for sd, evs in (tr.get("boosted_events") or {}).items()
               for (f, k) in evs}
    boosted_in_slices = 0
    if boosted:
        for s in slices:
            for f, tok in zip(s["grip_reconcile"].get("observed_frames", []),
                              s["grip_reconcile"]["observed_grippers"]):
                kind, _, arm = str(tok).partition("_")
                if (arm, int(f), kind) in boosted:
                    boosted_in_slices += 1
    row.update(
        compiles=compiles,
        slices=len(slices),
        matched_slices=sum(matched),
        observed_events=observed,
        zero_event_slices=zero_ev,
        label_contact_zero_events=label_zero,
        pinned_slices=pinned,
        # honest denominators (see above) — reported next to the originals
        slices_with_events=len(with_ev),
        matched_slices_strict=matched_strict,
        matched_slices_raw=matched_raw,
        tail_repaired_events=tail_ev,
        tail_repaired_slices=sum(1 for s in slices
                                 if s["grip_reconcile"].get("tail_repaired", 0)),
        boosted_events_in_slices=boosted_in_slices,
        observed_events_base=observed - boosted_in_slices,
        reconciled_raw=bool(compiles and slices and matched_raw == len(slices)),
        reconciled_strict=bool(compiles and with_ev
                               and matched_strict == len(with_ev)),
        prompt_events=tr.get("prompt_events", not tr.get("vanilla", False)),
        vanilla=tr.get("vanilla", False),
        grip_pins=len(lj.get("grip_pins", [])),
        valid_slices=sum(1 for s in slices if s["validation"]["ok"]),
        repairs=sum(s["validation"]["repairs"] for s in slices),
        skipped_slices=len(skipped),
        skipped_events=sum(s.get("skipped_events", 0) for s in skipped),
        skills_available=tr.get("skills", []),
        cost=tr.get("cost", {}),
        mismatches=mismatches,
        reconciled=bool(compiles and slices and all(matched)),
    )
    return row
