"""Stage A — one skeleton per label, from its representative episode's frames.

This is the only tier that spends model calls. It reuses the per-episode actor's
prompt machinery unchanged, with ONE deliberate difference: ``prog_so_far`` is
empty. A skeleton must be context-free to be reusable — code written to continue
one episode's program cannot be pasted into another's.
"""
from __future__ import annotations

import hashlib
import inspect
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from codegen import config
from codegen.actor.client import (MODEL, THROTTLE, USAGE, _strip_fences,
                                  call_model, load_key)
from codegen.actor.media import _to_pil, frame_b64, subtask_frames
from codegen.actor.prompts import CODER_SYS, CODER_SYS_VANILLA, CONV, build_coder
from codegen.actor.repair import repair_code
from codegen.api.active import API_DOC, EMBODIMENT
from codegen.grounding.facts import block_events
from codegen.library import SKILLS_DOC
from codegen.program.validate import validate_code
from codegen.skeleton.store import SkeletonStore, skeleton_key


def prompt_version(coder_model: str | None = None) -> str:
    """Everything that decides what a skeleton would look like, hashed.

    Same discipline as the per-episode cache key: prompt text, API surface, the
    frozen skill library, the builder's own source, and the knobs that choose the
    pixels. A change to any of them must not silently keep serving old code."""
    model = coder_model or MODEL
    gap = config.frame_gap(EMBODIMENT)
    stream = config.video_stream(EMBODIMENT)
    return hashlib.sha1("\n".join([
        model, CODER_SYS, CODER_SYS_VANILLA, CONV, API_DOC, SKILLS_DOC,
        inspect.getsource(build_coder),
        f"frame_gap={gap}", f"stream={stream}",
        f"max_slice_frames={config.get_int('MAX_SLICE_FRAMES')}",
    ]).encode()).hexdigest()[:12]


def generate_one(label: str, rep: dict, spec, apikey: str, coder_model: str,
                 repair_retries: int = 2, log=print):
    """Generate ONE skeleton from the representative occurrence. Returns a record."""
    episode = spec.load_episode(rep["task"], int(rep["episode"]))
    T = len(episode.sub_id)
    bs, be = int(rep["bs"]), min(int(rep["be"]), T)
    grips = block_events(episode.per_arm, bs, be)

    gap = config.frame_gap(EMBODIMENT)
    prompt_events = (EMBODIMENT == "yam")
    fr = subtask_frames(bs, be, gap, grips if prompt_events else [], T,
                        label_events=prompt_events,
                        max_frames=config.get_int("MAX_SLICE_FRAMES"))
    frame_items = []
    for name, fi in fr:
        img = _to_pil(episode.clip[min(int(fi), len(episode.clip) - 1)])
        frame_items.append((name, int(fi), frame_b64(img), ""))

    # CONTEXT-FREE: prog_so_far="" is what makes the result reusable elsewhere
    cmsgs, ctext = build_coder(frame_items, label, grips if prompt_events else [],
                               episode.instruction, prog_so_far="",
                               vanilla=not prompt_events)
    raw = call_model(cmsgs, apikey, coder_model,
                     max_tokens=config.get_int("VLM_MAX_TOKENS"),
                     temperature=config.get_float("VLM_TEMPERATURE"))
    code = _strip_fences(raw)
    problems = validate_code(code)
    repairs = 0
    while problems and repairs < repair_retries:
        repairs += 1
        raw = repair_code(cmsgs, "\n".join(code), problems, apikey, coder_model)
        code = _strip_fences(raw)
        problems = validate_code(code)
    if problems:
        log(f"[stage-a] {label[:48]!r}: still invalid after {repairs} repair(s): "
            f"{problems[0][:90]}")

    from codegen.program.effects import implied_grip_events
    implied = implied_grip_events(code) or []
    return {
        "label": label, "code": code,
        # the contract this skeleton will be matched on, everywhere it is reused
        "implied_events": [f"{k}_{a}" for (_off, a, k) in implied],
        "observed_events": [f"{k}_{a}" for (_f, a, k) in grips],
        "valid": not problems, "problems": problems[:4], "repairs": repairs,
        "n_frames": len(frame_items),
        "provenance": {"task": rep["task"], "episode": int(rep["episode"]),
                       "bs": bs, "be": be, "n_events": len(grips),
                       "instruction": episode.instruction},
    }


def _in_shard(label: str, shard: str) -> bool:
    """``i/N`` partitioning for MULTI-PROCESS stage A.

    Deterministic across processes on purpose: ``hash()`` is salted per
    interpreter, so using it here would give every worker a different partition
    and the fleet would both duplicate and miss labels."""
    if not shard:
        return True
    i, _, n = shard.partition("/")
    i, n = int(i), int(n)
    if not (0 <= i < n):
        raise ValueError(f"--shard {shard!r}: want i/N with 0 <= i < N")
    return int(hashlib.sha256(label.encode()).hexdigest(), 16) % n == i


def run(labels: dict, store: SkeletonStore, coder_model: str | None = None,
        limit: int = 0, log=print, parallel: int = 1, shard: str = "") -> dict:
    """Fill the store for every label that is missing one.

    Resumable by design: a label already present is skipped, so this can be
    killed and re-run. The work is I/O-bound (one model call per label, ~15s of
    it network), so ``parallel`` is a thread pool, not a process fleet.

    ACROSS processes, pass disjoint ``--shard i/N``. Sharing a store is safe
    (writes are atomic), but running two unsharded workers on the same inventory
    is NOT free: ``todo`` is snapshotted before the loop, so both would compute
    the same list and generate every label twice. The store is re-checked
    immediately before each call, which narrows that window but does not close
    it — the shard is what closes it."""
    from codegen.embodiments import get_spec
    spec = get_spec(EMBODIMENT)
    coder_model = coder_model or MODEL
    pv = prompt_version(coder_model)
    apikey = load_key()
    todo = [(lab, rec) for lab, rec in sorted(labels.items())
            if _in_shard(lab, shard)
            and store.get(skeleton_key(lab, pv, EMBODIMENT)) is None]
    if limit:
        todo = todo[:limit]
    parallel = max(int(parallel), 1)
    log(f"[stage-a] prompt_version={pv}  {len(labels)} labels, {len(todo)} missing"
        + (f", shard {shard}" if shard else "") + f", parallel={parallel}")
    made = failed = skipped = 0
    done = 0
    t0 = time.time()
    lock = threading.Lock()

    def one(item):
        label, rec = item
        key = skeleton_key(label, pv, EMBODIMENT)
        if store.get(key) is not None:      # another worker got here first
            return "skip", label, None
        try:
            record = generate_one(label, rec["representative"], spec, apikey, coder_model)
        except Exception as e:
            return "fail", label, f"{type(e).__name__}: {e}"
        record["prompt_version"] = pv
        record["coder_model"] = coder_model
        store.put(key, record)
        return "ok", label, None

    def account(res):
        nonlocal made, failed, skipped, done
        kind, label, err = res
        with lock:
            done += 1
            if kind == "ok":
                made += 1
            elif kind == "skip":
                skipped += 1
            else:
                failed += 1
                log(f"[stage-a] FAILED {label[:56]!r}: {err}")
            if done % 10 == 0 or done == len(todo):
                rate = done / max(time.time() - t0, 1e-9)
                log(f"[stage-a] {done}/{len(todo)}  made={made} failed={failed}"
                    + (f" skipped={skipped}" if skipped else "")
                    + f"  {time.time() - t0:.0f}s  ({rate * 60:.1f}/min)")

    if parallel == 1:
        for item in todo:
            account(one(item))
    else:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            for res in ex.map(one, todo):
                account(res)

    secs = time.time() - t0
    return {"labels": len(labels), "missing": len(todo), "made": made,
            "failed": failed, "skipped_taken_by_peer": skipped,
            "prompt_version": pv, "parallel": parallel, "shard": shard or None,
            "seconds": round(secs, 1),
            "seconds_per_label": round(secs / max(made, 1), 2),
            "model_calls": len(USAGE),
            "prompt_tokens": sum(u["prompt_tokens"] for u in USAGE),
            "completion_tokens": sum(u["completion_tokens"] for u in USAGE),
            "throttled": len(THROTTLE)}
