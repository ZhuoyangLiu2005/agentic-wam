"""Per-episode actor pipeline: load episode -> slice -> generate (continuing ONE
program) -> validate/repair -> ground to the gripper sensor -> write run-dir
artifacts (see codegen.pipeline.artifacts for the schema). The CLI shell
lives in scripts/run_episode.py.
"""
from __future__ import annotations

import base64
import hashlib
import inspect
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

from codegen import config
from codegen.actor.client import (MODEL, THROTTLE, USAGE, _strip_fences, call_model,
                                  load_key, model_short)
from codegen.actor.media import _to_pil, frame_b64, subtask_frames
from codegen.actor.prompts import CODER_SYS, CODER_SYS_VANILLA, CONV, build_coder
from codegen.actor.repair import repair_code
from codegen.api.active import API_DOC, EMBODIMENT
from codegen.embodiments import get_spec
from codegen.grounding.aperture import CONTACT_LABEL
from codegen.grounding.facts import block_events
from codegen.grounding.segment import sub_task_blocks, unmapped_ids
from codegen.library import SKILLS, SKILLS_DOC, skills_header
from codegen.pipeline import artifacts
from codegen.program.effects import match_grippers, repair_tail
from codegen.program.validate import validate_code


def _ml(s: str) -> list[str]:
    """Long prompt/output text -> list of lines (wrapped ~100 cols) so trace.json
    (indent=2) stays human-readable instead of one giant escaped-\\n string."""
    out_lines = []
    for ln in s.split("\n"):
        out_lines.extend(textwrap.wrap(ln, 100, drop_whitespace=False,
                                       replace_whitespace=False) or [""])
    return out_lines


def run_episode(args) -> int:
    _t_start = time.time()
    spec = get_spec(EMBODIMENT)
    episode = spec.load_episode(args.task, getattr(args, "episode", 0))
    instruction, clip = episode.instruction, episode.clip
    sub_id, text, per_arm = episode.sub_id, episode.text, episode.per_arm
    T = len(sub_id)
    apikey = load_key()
    # TWO distinct zero-label causes, mechanically separated (feedback §2.1 vs
    # §2.4): ids present in the data but missing from the map = the LOADING is
    # broken (selection layer without meta/) — always FATAL; no structure in
    # the data at all = a genuinely unannotated episode — always annotated as
    # one whole-episode slice, warned and recorded (below).
    bad_ids = unmapped_ids(sub_id, text)
    if bad_ids:
        print(f"[episode] FATAL {args.task}: {len(bad_ids)} sub-task id(s) present in "
              f"the per-frame data have NO entry in the id->text map "
              f"(ids {bad_ids[:8]}{'...' if len(bad_ids) > 8 else ''}; map has "
              f"{len(text)} entries) — the annotation MAPPING is broken or missing "
              f"(e.g. reading a selection layer without meta/). This is a "
              f"data-loading error, not a data property.", file=sys.stderr)
        return 3
    blocks = sub_task_blocks(sub_id, text, noise_labels=spec.noise_labels)
    unsegmented = not any(l for _, _, l, _n in blocks)
    if unsegmented:
        # ZERO labelled blocks with a HEALTHY mapping (the broken-mapping case
        # exited above) = a genuinely unannotated episode (~0.7% of the full
        # YAM corpus). Annotate it as ONE whole-episode slice — always allowed
        # by policy — but say so and record it, so the 15x-coarser structure is
        # visible in the log and filterable downstream (trace unsegmented=true).
        print(f"[episode] WARNING {args.task}: 0 labelled sub-task blocks "
              f"(T={T}, {len(text)} entries in the id->text map) — genuinely "
              f"unannotated; proceeding as ONE whole-episode slice "
              f"(trace.json records unsegmented=true).", file=sys.stderr)
        blocks = [(0, T, instruction, None)]

    # ---- build the ordered list of slices ---------------------------------------
    slices = []
    merged = []
    for (bs, be, label, noise) in blocks:
        if label:
            merged.append([bs, be, label, None, True])
        elif (merged and merged[-1][2] is None
                and merged[-1][4] == (noise is not None)):
            # coalesce only LIKE WITH LIKE. Merging an annotated-noise block into
            # an adjacent plain gap (or vice versa) used to DOWNGRADE the span to
            # non-noise, which silently sent the annotator's own 'mistake' /
            # 'reset both arms' spans to the VLM and wrote them into program.py as
            # task ground truth — measured at 38-44% of all noise blocks on the
            # real YAM corpus. Keeping the runs separate costs at most one extra
            # (cheap, unlabeled) slice and never loses the provenance.
            merged[-1][1] = be
            merged[-1][3] = merged[-1][3] or noise
        else:
            merged.append([bs, be, None, noise, noise is not None])
    for (bs, be, label, noise, all_noise) in merged:
        slices.append({"bs": bs, "be": be, "label": label, "header": True,
                       "kind": "subtask", "noise": noise if all_noise else None})

    # output dir: layout knowledge lives in the embodiment spec, not here, and
    # the model->path-segment rule lives in codegen.actor.client (the batch gate
    # derives the same segment from the same function).
    out = spec.run_dir(args.task, args.tag, model_short(args.coder_model),
                       is_default_model=(args.coder_model == MODEL),
                       episode_id=episode.episode_id)
    trace_dir = out / artifacts.TRACE_DIR
    trace_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out / artifacts.CACHE
    cache = json.loads(cache_path.read_text()) if (cache_path.exists() and not args.no_cache) else {}
    # Actor-input mode. --vanilla: annotation-only prompt AND uniform sampling
    # (pure ablation). EGO default: events NEVER reach the prompt (weak
    # aperture proxy under- and mis-detects contacts heavily);
    # they still ANCHOR frame sampling, neutrally named. YAM default keeps the
    # sensor-authoritative event prompt. Grounding/reconciliation below always
    # runs against the observed events — with prompt_events False it is an
    # INDEPENDENT post-hoc diagnostic, and a fully-matched slice still earns
    # its fine-grained grip_pins.
    vanilla = bool(getattr(args, "vanilla", False))
    prompt_events = (EMBODIMENT == "yam") and not vanilla
    anchor_events = not vanilla
    # Everything that changes WHAT THE MODEL SEES must be in this key. frame_gap
    # and the video stream are not prompt TEXT but they decide the pixels: without
    # them a re-run at a different gap (or camera) reuses every cached generation
    # while rewriting trace.json's frame list and the trace/*.jpg audit copies —
    # the run dir then claims the model saw frames it never saw.
    video_stream = config.video_stream(EMBODIMENT)
    prompt_key = hashlib.sha1("\n".join([
        args.coder_model, CODER_SYS if prompt_events else CODER_SYS_VANILLA, CONV, API_DOC,
        SKILLS_DOC, inspect.getsource(build_coder),
        f"vanilla={vanilla}", f"prompt_events={prompt_events}",
        f"frame_gap={int(args.frame_gap)}", f"video_stream={video_stream}",
        f"max_slice_frames={config.get_int('MAX_SLICE_FRAMES')}",
    ]).encode()).hexdigest()[:12]

    def cached(kind, k, fn):
        ck = f"{kind}_{prompt_key}:{k}"
        if ck in cache:
            return cache[ck]
        v = fn()
        cache[ck] = v
        cache_path.write_text(json.dumps(cache, indent=1))
        return v

    def cache_put(kind, k, v):   # overwrite an entry (a repaired coder output)
        cache[f"{kind}_{prompt_key}:{k}"] = v
        cache_path.write_text(json.dumps(cache, indent=1))

    model_line = f'# model: vlm (perceive+code, one call)={args.coder_model}'
    gen_note = ["# Auto-generated by scripts/run_episode.py: per slice ONE VLM reads the",
                "#   frames AND writes the aligned code in a single call.",
                "#   Full per-slice VLM inputs+outputs saved in trace.json."]
    header = [
        f'# {spec.display_name} (VLM-in-the-loop, unit=subtask) :: "{instruction}"',
        model_line,
        *gen_note,
        "import numpy as np",
    ]
    header += skills_header(SKILLS)   # composed-function skills, callable below
    lines = list(header)
    max_slice_frames = config.get_int("MAX_SLICE_FRAMES")
    capped_slices = 0                 # slices that hit MAX_SLICE_FRAMES (see media._cap)
    frame_src_block = [None] * T      # per-frame: [first_line, last_line] of the sub-task block
    rec = []
    fmap = []          # (s, e, [line_nums]) per slice
    grip_pins = []     # [frame, line] exact gripper transition->line pins (matched slices only)
    trace = []
    done = 0

    for i, sl in enumerate(slices):
        if args.max_slices and done >= args.max_slices:
            break
        bs, be = sl["bs"], sl["be"]
        if sl["header"]:
            lines.append("")
            lines.append(f"# --- {sl['label'] or '(unlabeled transition)'}  (frames {bs}-{be})")

        grips = block_events(per_arm, bs, be)

        # §2.5: two skip rules for UNLABELED spans, distinguished by WHY the
        # span is unlabeled (segment.sub_task_blocks provenance):
        #   * annotated NOISE ("reset both arms"/"mistake"/...): the annotator
        #     SAID this span is not task content — trust the annotation and
        #     skip ALWAYS, events included. Coding a mistake segment would
        #     manufacture wrong GT. Its observed events are excluded from
        #     reconciliation and recorded (trace skipped_events; gate reports
        #     them per task) — excluded loudly, never silently dropped.
        #   * plain GAP (nobody annotated anything): skip only when ZERO
        #     events — an event here may be real task content, so it must be
        #     grounded by the VLM as before.
        # Labeled zero-event slices are NEVER skipped (annotation is GT; its
        # motion must be coded).
        noise = sl.get("noise")
        if sl["label"] is None and (noise is not None or not grips):
            why = (f"noise: '{noise}'" if noise is not None else "unlabeled, 0 events")
            tail = (f" — {len(grips)} observed event(s) excluded from grounding"
                    if grips else " — no observed events")
            lines[-1] += f" [{why}] skipped, no VLM call{tail}"
            lns = [len(lines)]
            fmap.append((bs, be, lns))
            grip_report = {
                "coder_grippers": [],
                "observed_grippers": [f"{k}_{a}" for (_f, a, k) in grips],
                "observed_frames": [int(_f) for (_f, _a, _k) in grips],
                "matched": None,          # not scored — excluded, not passed
            }
            f = int(np.clip(be - 1, 0, T - 1))
            rec.append({"label": None, "kind": sl["kind"], "start": bs, "end": be,
                        "frames": [], "vlm": f"(skipped: {why})",
                        "code": [], "grip_reconcile": grip_report,
                        "logged_left_pos": np.round(per_arm["left"]["pos"][f], 4).tolist(),
                        "logged_right_pos": np.round(per_arm["right"]["pos"][f], 4).tolist()})
            trace.append({
                "slice": i, "label": None, "kind": sl["kind"], "start": bs, "end": be,
                "skipped": True, "noise_label": noise,
                "skipped_events": len(grips), "frames": [],
                "grip_reconcile": grip_report,
                "code": {"lines": []},
                "validation": {"ok": True, "problems": [], "repairs": 0}})
            print(f"  [{i}] {sl['kind']:8s} f{bs}-{be}: skipped ({why}{tail})",
                  file=sys.stderr)
            continue          # deliberately NOT counted against --max-slices

        fr = subtask_frames(bs, be, args.frame_gap, grips if anchor_events else [],
                            T, label_events=prompt_events,
                            max_frames=max_slice_frames)
        # a slice sitting exactly ON the ceiling is reported as capped: the
        # request would otherwise have been larger than the router accepts
        slice_capped = bool(max_slice_frames and len(fr) >= max_slice_frames)
        capped_slices += int(slice_capped)

        # frames sent to the VLM (endpoints + a pre frame just before each gripper
        # event). The trace copy on disk is the EXACT JPEG payload the model
        # receives — byte-identical audit material, and ~5.5x smaller than the
        # PNG re-encode it replaces.
        frame_items = []
        for name, fi in fr:
            img = _to_pil(clip[min(int(fi), len(clip) - 1)])
            safe = name.replace("/", "_").replace("@", "at")
            rel = f"{artifacts.TRACE_DIR}/slice{i:02d}_{safe}_f{int(fi)}.jpg"
            b64 = frame_b64(img)
            (out / rel).write_bytes(base64.b64decode(b64.split(",", 1)[1]))
            frame_items.append((name, int(fi), b64, rel))

        # program-so-far (all earlier subtasks, incl. this slice's header comment) -> the
        # coder CONTINUES one coherent program. Its hash goes in the cache key so a change
        # to any earlier slice regenerates the later ones.
        prog_so_far = "\n".join(lines[len(header):])
        prog_key = hashlib.sha1(prog_so_far.encode()).hexdigest()[:8]
        slice_key = f"{bs}:{be}:{prog_key}"
        desc = "(perception folded into the coder call — no separate description)"
        cmsgs, ctext = build_coder(frame_items, sl["label"],
                                   grips if prompt_events else [],
                                   instruction, prog_so_far=prog_so_far,
                                   vanilla=not prompt_events)
        ckind = "codevis"
        raw = cached(ckind, slice_key,
                     lambda: call_model(cmsgs, apikey, args.coder_model,
                                        max_tokens=config.get_int("VLM_MAX_TOKENS"),
                                        temperature=config.get_float("VLM_TEMPERATURE")))
        code = _strip_fences(raw)

        # ---- VALIDATION stage (static): the slice must compile in program context and only
        # call real API functions. On failure the SAME model gets its output + the problems
        # back and rewrites the slice; a successful repair replaces the cached output. If it
        # STILL doesn't parse, the block is commented out so program.py always compiles
        # (alignment is kept — the slice's frames map to the commented block).
        problems = validate_code(code, prog_so_far)
        val = {"ok": not problems, "problems": problems, "repairs": 0}
        while problems and val["repairs"] < args.repair_retries:
            print(f"      validation FAILED ({problems[0]}{' ...' if len(problems) > 1 else ''})"
                  f" -> repair round {val['repairs'] + 1}", file=sys.stderr)
            raw = repair_code(cmsgs, raw, problems, apikey, args.coder_model)
            code = _strip_fences(raw)
            problems = validate_code(code, prog_so_far)
            val["repairs"] += 1
            val["ok"], val["problems"] = not problems, problems
        if val["repairs"] and val["ok"]:
            cache_put(ckind, slice_key, raw)
        if any(p.startswith("SyntaxError") for p in problems):
            code = [f"# INVALID SLICE (failed validation after {val['repairs']} repairs) — "
                    f"kept commented out for alignment:"] + [f"# {ln}" for ln in code]
            val["commented_out"] = True

        # Compare the coder's grippers to the observed signal. Alignment is block-level, but
        # when the coder's grippers MATCH the observed events we still get an EXACT
        # gripper-transition-frame -> gripper-line correspondence (grip_anchors); we keep
        # those to mark the precise gripper moments. A mismatch yields no anchors.
        grip_anchors, grip_report = match_grippers(code, grips)
        matched_raw = grip_report["matched"]        # BEFORE any deterministic repair
        tail_repaired = 0
        if not matched_raw and config.get_bool("REPAIR_TAIL"):
            code, tail_repaired = repair_tail(code, grips)
            if tail_repaired:
                grip_anchors, grip_report = match_grippers(code, grips)
                print(f"      tail-repaired: +{tail_repaired} primitive(s) from the sensor "
                      f"-> matched", file=sys.stderr)
        grip_report["matched_raw"] = matched_raw    # the model's own score, kept
        grip_report["tail_repaired"] = tail_repaired
        if not grip_report["matched"]:
            print(f"      grippers != observed GT (diagnostic): coder "
                  f"{grip_report['coder_grippers']} vs {grip_report['observed_grippers']}",
                  file=sys.stderr)

        lns = []
        for cl in code:
            lines.append(cl)
            lns.append(len(lines))
        if not lns:
            lines.append(f"# (model produced no code for slice {i})")
            lns = [len(lines)]
        fmap.append((bs, be, lns))
        for (off, gf) in grip_anchors:            # exact gripper frame -> program line
            if 0 <= off < len(lns) and 0 <= gf < T:
                grip_pins.append([int(gf), int(lns[off])])

        f = int(np.clip(be - 1, 0, T - 1))
        rec.append({"label": sl["label"], "kind": sl["kind"], "start": bs, "end": be,
                    "frames": [it[1] for it in frame_items], "vlm": desc, "code": code,
                    "grip_reconcile": grip_report,
                    "logged_left_pos": np.round(per_arm["left"]["pos"][f], 4).tolist(),
                    "logged_right_pos": np.round(per_arm["right"]["pos"][f], 4).tolist()})
        trace.append({
            "slice": i, "label": sl["label"], "kind": sl["kind"], "start": bs, "end": be,
            "frames": [{"name": n, "t": t, "png": rel} for (n, t, _b, rel) in frame_items],
            # sampling provenance + the §4.1 flag (contact-verb label, silent detector)
            "frame_anchors": "events" if (anchor_events and grips) else "uniform",
            "frames_capped": slice_capped,
            "label_contact_zero_events": bool(
                sl["label"] and CONTACT_LABEL.search(sl["label"]) and not grips),
            "grip_reconcile": grip_report,
            "perceive": {"note": "folded into the coder call (single VLM sees frames + writes code)"},
            "code": {"system": _ml(cmsgs[0]["content"]),
                     "input": _ml(ctext), "output_raw": _ml(raw), "lines": code},
            "validation": val,
        })
        print(f"  [{i}] {sl['kind']:8s} f{bs}-{be}: {desc[:72]}", file=sys.stderr)
        done += 1

    # frame map: BLOCK-level. Every frame of a slice maps to that slice's WHOLE code block
    # (all the lines it emitted) as a line RANGE (frame_src_block = [first_line, last_line]).
    for (s, e, lns) in sorted(fmap, key=lambda x: x[0]):
        lo, hi = min(lns), max(lns)
        for t in range(s, e):
            frame_src_block[t] = [lo, hi]
    # fill uncovered frames (dropped trivial gaps) forward, then back-fill the lead
    last_block = None
    for t in range(T):
        if frame_src_block[t] is None:
            frame_src_block[t] = last_block
        else:
            last_block = frame_src_block[t]
    first_block = next((b for b in frame_src_block if b is not None), None)
    for t in range(T):
        if frame_src_block[t] is None:
            frame_src_block[t] = first_block
        else:
            break

    (out / artifacts.PROGRAM).write_text("\n".join(lines) + "\n")
    (out / artifacts.LINES).write_text(
        json.dumps({"program": artifacts.PROGRAM,
                    "frame_src_block": frame_src_block, "grip_pins": sorted(grip_pins)}))
    (out / artifacts.CHUNKS).write_text(
        json.dumps({"instruction": instruction, "T": int(T),
                    "unit": "subtask", "phases": rec}, indent=2))
    (out / artifacts.TRACE).write_text(json.dumps(
        {"task": args.task, "instruction": instruction, "unit": "subtask",
         "coder_model": args.coder_model,
         # actor-input mode: vanilla = annotation-only prompt AND uniform
         # sampling; prompt_events=False (all ego runs + vanilla) = no
         # detector content in the prompt, so grip_reconcile is an INDEPENDENT
         # post-hoc diagnostic (and the admission-evidence criterion)
         "vanilla": vanilla, "prompt_events": prompt_events,
         # WHICH PIXELS the actor read: the sampling gap, the per-slice cap, and
         # the video stream (code-GT is a HIGH-RESOLUTION contract — a run made
         # from the 320x240 pretrain stream must be identifiable as such).
         "frame_gap": int(args.frame_gap),
         # what this episode actually COST — wall clock and tokens, the two
         # numbers a campaign is planned from and neither recoverable later
         "cost": {"model_calls": len(USAGE),
                  "seconds_in_model": round(sum(u["seconds"] for u in USAGE), 1),
                  "wall_seconds": round(time.time() - _t_start, 1),
                  "prompt_tokens": sum(u["prompt_tokens"] for u in USAGE),
                  "completion_tokens": sum(u["completion_tokens"] for u in USAGE),
                  # what the router pushed back with, and what it told us about why
                  "throttled": len(THROTTLE),
                  "throttle_seconds": round(sum(t["sleep"] for t in THROTTLE), 1),
                  "throttle_sample": THROTTLE[:3]},
         "max_slice_frames": config.get_int("MAX_SLICE_FRAMES"),
         "video_stream": video_stream,
         "capped_slices": capped_slices,
         # True = zero labelled blocks; annotated as ONE whole-episode slice
         # under --allow-unsegmented. Downstream filters on this flag.
         "unsegmented": unsegmented,
         # detector provenance: events ADDED by the annotation-conditioned second
         # pass (empty for embodiments/detectors without one) — audit material
         "boosted_events": {sd: [list(e) for e in (per_arm[sd].get("boosted") or [])]
                            for sd in ("left", "right")},
         "skills": [s["name"] for s in SKILLS],
         "n_slices": len(trace), "slices": trace}, indent=2))

    # whole-program compile check — must always pass (unparseable slices were commented out)
    try:
        compile("\n".join(lines), str(out / artifacts.PROGRAM), "exec")
        prog_ok = True
    except SyntaxError as e:
        prog_ok = False
        print(f"  WARNING: assembled program.py does NOT compile: line {e.lineno}: {e.msg}",
              file=sys.stderr)

    covered = sum(1 for b in frame_src_block if b is not None)
    n_val_ok = sum(1 for t in trace if t["validation"]["ok"])
    n_repairs = sum(t["validation"]["repairs"] for t in trace)
    print(f"{args.task}: unit=subtask "
          f"slices={len(slices)} model_calls={done} lines={len(lines)} "
          f"covered={covered}/{T} valid={n_val_ok}/{len(trace)} repairs={n_repairs} "
          f"program_compiles={prog_ok} -> {out}")
    return 0
