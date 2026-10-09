#!/usr/bin/env python
"""Batch runner + acceptance gate for the YAM inverse-codegen data-gen pipeline.

The ASPIRE "actor fleet" step — one round of the SYNCHRONOUS BATCH LOOP: every
episode in the batch runs in PARALLEL (--parallel), one independent actor each
(scripts/run_episode.py); actors are identical and cannot communicate (own run
dir + cache; the skill library is a frozen read-only snapshot for the whole batch).
After the batch, the coordinator audits and grows the library; the NEXT batch's
actors load the updated library at startup. Each finished episode is scored with the
per-task RECONCILIATION METRIC (codegen.pipeline.gate):

    reconciled = program.py compiles
                 AND every slice's gripper reconciliation matched
                     (every observed gripper transition pinned to its producing line)

This is an EVAL PROXY for comparing pipeline versions (prompts, library states) —
NOT a dataset-inclusion decision: gripper-event alignment is
itself a proxy, so every episode's program remains a dataset candidate regardless
of the verdict (unreconciled slices simply fall back to block-level alignment).
The coordinator (.claude/yam/main-agent-prompt.md) reads the manifest this
writes, audits the programs (reconciled runs carry the strongest admission
evidence), and admits composed-function skills to skill_library/.

Run from the workspace root (PY=.venv/bin/python):

    $PY scripts/run_batch.py --embodiment ego --tag v1 --tasks a,b   # one shard

The --tag is the PIPELINE VERSION, shared by every batch of a campaign; each
dispatch auto-assigns the next --round shard index (bookkeeping only — run dirs
<task>/<model>/<episode>/<tag>/ carry the version alone, and single-pass means
each task is annotated once per version). The ONLY state that carries between
rounds is the skill library (skill_library/, loaded by the actor at startup, so
whatever the coordinator admitted last round is injected automatically).
--no-run re-applies the gate to an existing batch (pass its --round).

Outputs under outputs/<emb>/batch_<tag>_<round>/:
    manifest.jsonl   one row per task (gate verdict + counters)
    summary.json     aggregate (acceptance rate, failure reasons)
    <task>.log       full codegen stderr/stdout per task
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codegen.pipeline.gate import ROOT, gate_task  # noqa: E402
from codegen.skill_paths import LIBRARY_ROOT  # noqa: E402


def main() -> int:
    from codegen import config
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default=config.get("TAG"),
                    help="pipeline-VERSION tag, shared by every batch of one campaign: "
                         "run dirs -> <task>/<model>/<episode>/<tag>/ (single-pass: each "
                         "task is annotated once per tag). Default: the `tag` config key")
    ap.add_argument("--round", default="",
                    help="batch shard label within the campaign (batch outputs -> "
                         "outputs/<emb>/batch_<tag>_<round>/). Default: AUTO — the next "
                         "free rN for this tag. The round exists ONLY in this bookkeeping "
                         "path; run dirs and the dataset carry the version tag alone. "
                         "Required with --no-run (a re-gate targets an existing batch).")
    ap.add_argument("--tasks", default="", help="comma-separated subset (default: all)")
    ap.add_argument("--tasks-file", default="",
                    help="file of work units, one per line: '<task>' or "
                         "'<task>:<episode_index>' ('#' comments allowed). The "
                         "scalable form — argv caps a comma list at ~131 KB")
    ap.add_argument("--skip-done", action="store_true",
                    help="resume: skip units whose run dir already carries "
                         "program.py + trace.json (they are re-GATED into the "
                         "manifest, no actor call); pairs with --round to "
                         "continue an interrupted batch")
    ap.add_argument("--parallel", type=int, default=config.get_int("PARALLEL"),
                    help="concurrent actors (episodes) within the batch. Actors are "
                         "independent and identical by construction (own run dir + cache, "
                         "shared READ-ONLY library) — the ceiling is the VLM router's rate "
                         "limit, not the machine")
    ap.add_argument("--no-run", action="store_true",
                    help="gate existing run dirs only; never invoke the actor")
    ap.add_argument("--vanilla", action="store_true", default=None,
                    help="forwarded to every actor: full detector-influence "
                         "ablation (see run_episode --vanilla). Default: the "
                         "`vanilla` config key (flag present = force on). Use a "
                         "dedicated tag so vanilla and standard runs never share "
                         "run dirs")
    ap.add_argument("--embodiment", default=config.embodiment(),
                    choices=["yam", "ego", "tianji_sharpa", "yam_sft"],
                    help="which embodiment's data/API/library this batch annotates "
                         "(default: CODEGEN_EMBODIMENT env / the config file)")
    ap.add_argument("--coder-model", default=config.get("CODER_MODEL"),
                    help="the actor VLM. Forwarded to every actor AND used to "
                         "derive the run-dir segment the gate reads — the two "
                         "must agree, so never let the actor pick a different one")
    args = ap.parse_args()
    if args.vanilla is None:
        args.vanilla = config.get_bool("VANILLA")
    if not args.tag:
        print("[batch] no version tag: set `tag:` in your config file "
              "(CODEGEN_CONFIG=configs/<file>.yaml) or pass --tag", file=sys.stderr)
        return 1
    os.environ["CODEGEN_EMBODIMENT"] = args.embodiment
    from codegen.embodiments import get_spec  # noqa: PLC0415 (after env is set)
    spec = get_spec(args.embodiment)

    def parse_unit(tok: str) -> tuple[str, int]:
        task, _, idx = tok.partition(":")
        return task.strip(), int(idx) if idx.strip() else 0

    units: list[tuple[str, int]] = []
    if args.tasks_file:
        for line in Path(args.tasks_file).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                units.append(parse_unit(line))
    if args.tasks:
        units += [parse_unit(t) for t in args.tasks.split(",") if t.strip()]
    known = set(spec.list_tasks())
    if not units:
        units = [(t, 0) for t in sorted(known)]
    missing = sorted({t for t, _ in units} - known)
    if missing:
        print(f"[batch] unknown tasks: {missing}", file=sys.stderr)
        return 1

    if args.no_run and not args.round:
        print("[batch] --no-run re-gates an EXISTING batch: pass its --round", file=sys.stderr)
        return 1
    rid = args.round
    if not rid:                       # AUTO: next free shard index for this tag
        base = spec.batch_dir("probe").parent
        taken = [int(m.group(1)) for d in base.glob(f"batch_{args.tag}_r*")
                 for m in [re.fullmatch(re.escape(f"batch_{args.tag}_r") + r"(\d+)", d.name)] if m]
        rid = f"r{max(taken, default=0) + 1}"
    out = spec.batch_dir(f"{args.tag}_{rid}")
    out.mkdir(parents=True, exist_ok=True)
    # the batch writes its own driver.log (the auto-assigned dir isn't known to
    # the launching shell, so an external redirect can't target it)
    log_f = (out / "driver.log").open("a")

    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s); st.flush()
            return len(s)
        def flush(self):
            for st in self.streams:
                st.flush()
    sys.stderr = _Tee(sys.__stderr__, log_f)
    sys.stdout = _Tee(sys.__stdout__, log_f)
    _t_batch = time.time()
    print(f"[batch] tag={args.tag} round={rid} -> {out}", file=sys.stderr)

    from codegen import config as _cfg
    from codegen.actor.client import model_short          # noqa: PLC0415 (after env)
    timeout_s = _cfg.get_int("ACTOR_TIMEOUT")
    mshort = model_short(args.coder_model)   # the run-dir segment the actor writes

    def unit_label(t, ei):
        return t if ei == 0 else f"{t}:{ei}"

    def process(unit: tuple[str, int]) -> dict:
        """One actor: run the episode's codegen, then gate it. Actors within the
        batch are identical and share NOTHING writable (own run dir + cache; the
        skill library is a frozen read-only snapshot for the whole batch)."""
        t, ei = unit
        eid = f"episode_{ei:03d}"
        name = unit_label(t, ei)
        logname = name.replace(":", "_ep")
        rd = spec.run_dir(t, args.tag, mshort, episode_id=eid)
        if args.skip_done and (rd / "program.py").exists() and (rd / "trace.json").exists():
            row = gate_task(t, rd)
            row["resumed"] = True
            print(f"[batch] {name}: already done — re-gated, actor skipped", file=sys.stderr)
            return row
        if not args.no_run:
            err = (spec.preflight(t, ei)
                   if "episode_index" in spec.preflight.__code__.co_varnames
                   else spec.preflight(t))
            if err:
                print(f"[batch] {name}: SKIP ({err})", file=sys.stderr)
                return {"task": t, "episode": eid,
                        "run_dir": str(rd.relative_to(ROOT)), "reconciled": False,
                        "error": err}
            cmd = [sys.executable, "scripts/run_episode.py", "--task", t,
                   "--episode", str(ei),
                   "--tag", args.tag, "--embodiment", args.embodiment,
                   "--coder-model", args.coder_model]
            if args.vanilla:
                cmd.append("--vanilla")
            try:
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                   timeout=timeout_s)
            except subprocess.TimeoutExpired as e:
                (out / f"{logname}.log").write_text(
                    str(e.stdout or "") + str(e.stderr or ""))
                print(f"[batch] {name}: ACTOR TIMEOUT after {timeout_s}s", file=sys.stderr)
                return {"task": t, "episode": eid,
                        "run_dir": str(rd.relative_to(ROOT)), "reconciled": False,
                        "error": f"actor timeout after {timeout_s}s (re-queue candidate)"}
            (out / f"{logname}.log").write_text(r.stdout + r.stderr)
            if r.returncode != 0:
                print(f"[batch] {name}: ACTOR FAILED (exit {r.returncode})", file=sys.stderr)
                return {"task": t, "episode": eid,
                        "run_dir": str(rd.relative_to(ROOT)), "reconciled": False,
                        "error": f"actor exit {r.returncode} (see {logname}.log)"}
        row = gate_task(t, rd)
        verdict = "RECONCILED" if row.get("reconciled") else "partial"
        print(f"[batch] {name}: {verdict} "
              f"(slices {row.get('matched_slices', '?')}/{row.get('slices', '?')} matched, "
              f"pins {row.get('grip_pins', '?')}/{row.get('observed_events', '?')}, "
              f"compiles={row.get('compiles')})", file=sys.stderr)
        return row

    # INCREMENTAL manifest: one row appended the moment each unit finishes, so a
    # batch that dies mid-run keeps every completed record (resume: same --round
    # plus --skip-done). Duplicate (task, episode) rows across resumes are
    # deduped below, LAST occurrence wins.
    rows = []
    mf = (out / "manifest.jsonl").open("a")
    with ThreadPoolExecutor(max_workers=max(args.parallel, 1)) as ex:
        futs = {ex.submit(process, u): u for u in units}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r0 = fut.result()
            except Exception as e:
                # One unreadable/half-written run dir must not take the shard
                # with it: gate_task raising here used to propagate out of the
                # loop, abandoning every unit still queued AND every completed
                # row not yet flushed. Record it as an errored unit instead —
                # gen_progress re-queues errored units, so it comes back.
                t, ei = futs[fut]
                eid = f"episode_{ei:03d}"
                print(f"[batch] {unit_label(t, ei)}: GATE FAILED "
                      f"({type(e).__name__}: {e})", file=sys.stderr)
                r0 = {"task": t, "episode": eid,
                      "run_dir": str(spec.run_dir(t, args.tag, mshort,
                                                  episode_id=eid).relative_to(ROOT)),
                      "reconciled": False,
                      "error": f"gate failed: {type(e).__name__}: {str(e)[:160]}"}
            rows.append(r0)
            mf.write(json.dumps(r0) + "\n")
            mf.flush()
            print(f"[batch] progress {i}/{len(units)}", file=sys.stderr)
    mf.close()

    last = {}
    for line in (out / "manifest.jsonl").read_text().splitlines():
        if line.strip():
            r0 = json.loads(line)
            last[(r0["task"], r0.get("episode", "episode_000"))] = r0
    rows = sorted(last.values(), key=lambda r0: (r0["task"], r0.get("episode", "")))
    with (out / "manifest.jsonl").open("w") as f:
        for r0 in rows:
            f.write(json.dumps(r0) + "\n")
    n_rec = sum(1 for r0 in rows if r0.get("reconciled"))
    total_obs = sum(r0.get("observed_events", 0) for r0 in rows)
    total_pins = sum(r0.get("grip_pins", 0) for r0 in rows)
    n_slices = sum(r0.get("slices", 0) for r0 in rows)
    summary = {
        "tag": args.tag, "round": rid, "tasks": len(rows),
        "skipped_events": sum(r0.get("skipped_events", 0) for r0 in rows),
        "skills_in_library": sorted({s for r0 in rows for s in r0.get("skills_available", [])}),
        # actor-input mode of the batch (rows should be homogeneous):
        # prompt_events=False -> reconcile/pin are an INDEPENDENT diagnostic
        # (detector-vs-VLM disagreement), not a quality headline
        "prompt_events": all(r0.get("prompt_events", True) for r0 in rows),
        # EVAL METRICS (pipeline-version comparison — NOT dataset-inclusion decisions)
        "fully_reconciled": n_rec,
        "reconcile_rate": round(n_rec / len(rows), 3) if rows else 0.0,
        "pin_rate": round(total_pins / total_obs, 3) if total_obs else 0.0,
        # detector-pressure + pin-coverage headlines (fractions of coded slices):
        # zero-event slices carry no reconciliation signal at all;
        # label_contact_zero_events = contact-verb label with a silent detector
        # (feedback §4.1); pinned_slice_rate = slices that EARNED fine-grained
        # frame->line pins (the downstream alignment-value metric)
        "zero_event_slice_rate": round(sum(r0.get("zero_event_slices", 0) for r0 in rows)
                                       / n_slices, 3) if n_slices else 0.0,
        "label_contact_zero_events": sum(r0.get("label_contact_zero_events", 0) for r0 in rows),
        "pinned_slice_rate": round(sum(r0.get("pinned_slices", 0) for r0 in rows)
                                   / n_slices, 3) if n_slices else 0.0,
        # --- the same three numbers with honest denominators -------------------
        # STRICT ignores zero-event slices (empty == empty is not evidence);
        # RAW is before deterministic tail repair (the model's own score);
        # BASE removes the annotation-boosted events (ego), which are not
        # independent of the annotation the actor was shown.
        "reconcile_rate_strict": round(
            sum(1 for r0 in rows if r0.get("reconciled_strict")) / len(rows), 3
        ) if rows else 0.0,
        "reconcile_rate_raw": round(
            sum(1 for r0 in rows if r0.get("reconciled_raw")) / len(rows), 3
        ) if rows else 0.0,
        "matched_slice_rate_strict": round(
            sum(r0.get("matched_slices_strict", 0) for r0 in rows)
            / sum(r0.get("slices_with_events", 0) for r0 in rows), 3
        ) if sum(r0.get("slices_with_events", 0) for r0 in rows) else 0.0,
        "pin_rate_base": round(
            total_pins / sum(r0.get("observed_events_base", 0) for r0 in rows), 3
        ) if sum(r0.get("observed_events_base", 0) for r0 in rows) else 0.0,
        "boosted_event_share": round(
            sum(r0.get("boosted_events_in_slices", 0) for r0 in rows) / total_obs, 3
        ) if total_obs else 0.0,
        "tail_repaired_events": sum(r0.get("tail_repaired_events", 0) for r0 in rows),
        "tail_repaired_slices": sum(r0.get("tail_repaired_slices", 0) for r0 in rows),
        # units that produced NO program (actor crash/timeout, preflight skip,
        # gate failure). gen_progress re-queues exactly these, so the count here
        # and the "re-queued" line there must agree.
        # campaign planning numbers, summed over the round
        "cost": {
            k: sum((r0.get("cost") or {}).get(k, 0) for r0 in rows)
            for k in ("model_calls", "seconds_in_model", "prompt_tokens",
                      "completion_tokens", "throttled", "throttle_seconds")
        } | {"wall_seconds_sum": round(sum((r0.get("cost") or {}).get("wall_seconds", 0)
                                           for r0 in rows), 1),
             "wall_seconds_batch": round(time.time() - _t_batch, 1),
             "episodes_costed": sum(1 for r0 in rows if r0.get("cost"))},
        "errored": sum(1 for r0 in rows if r0.get("error")),
        "error_kinds": dict(collections.Counter(
            str(r0["error"]).split("(")[0].strip()
            for r0 in rows if r0.get("error"))),
        "partially_reconciled": [
            {"task": r0["task"], "episode": r0.get("episode", "episode_000"),
             "why": r0.get("error")
                    or (None if r0.get("compiles") else "does not compile")
                    or f"{r0['slices'] - r0['matched_slices']}/{r0['slices']} slices unreconciled"}
            for r0 in rows if not r0.get("reconciled")],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # Freeze this round's library state next to its results (ASPIRE frozen-
    # snapshot convention): the batch ran against EXACTLY this library, so the
    # round stays reproducible/comparable after the library grows.
    if not args.no_run:
        # The snapshot records the library the batch RAN AGAINST — a --no-run
        # re-gate must never overwrite it with today's library.
        snap = out / "library_snapshot"
        if snap.exists():
            shutil.rmtree(snap)      # same-tag re-RUN refreshes the snapshot
        for label, src in (("skills", spec.skills_dir),
                           ("patterns", LIBRARY_ROOT / "patterns")):
            if src.is_dir():
                shutil.copytree(src, snap / label)

    print(f"[batch] {n_rec}/{len(rows)} fully reconciled, "
          f"pin rate {summary['pin_rate']:.1%} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
