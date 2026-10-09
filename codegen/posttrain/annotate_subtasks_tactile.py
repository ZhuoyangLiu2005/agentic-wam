"""Stage 0 for downstream Sharpa data: TACTILE SPLITS, THEN A VLM NAMES THE SPANS.

Pipeline A slices an episode with `segment.sub_task_blocks(sub_id, text)` — per-frame sub-task ids
plus a label per id. On the pretrain corpora those come from HUMAN ANNOTATION. Downstream real-robot
data has none, and that is the only thing pipeline A is missing here.

So this script manufactures exactly that one artifact, and nothing else:

  1. SPLIT with tactile.  Boundaries are contact transitions (opposition rule: thumb AND
     (index OR middle)), merged across both hands. A direct sensor decides WHERE the sub-tasks
     change — not a VLM, which would be guessing at boundaries it cannot see precisely.
  2. NAME with the VLM.   One cheap call per span: frames in, a short imperative label out.
     The VLM only has to answer "what is happening here", never "where does it start".

Output is a per-episode `sub_id` (T,) int32 + `text` {id: label} — byte-compatible with what the
annotated corpora provide, so `sub_task_blocks` and everything downstream runs UNCHANGED.

Why not let the VLM do both: boundary precision is what the event contract grades against. A label
that starts 20 frames late shifts the implied-event window and the program stops reconciling. The
tactile edge is frame-accurate; a VLM's is not.

Span hygiene mirrors the cap-x window builder: drop boundaries that leave < MIN_SPAN_S on either
side, and subdivide anything longer than MAX_SPAN_S so one label never covers two sub-tasks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

MIN_SPAN_S = 1.0
MAX_SPAN_S = 8.0
FRAMES_PER_SPAN = 4

PROMPT = """You are labelling one segment of a robot manipulation episode.

Overall task: {instruction}

The frames below are consecutive moments from ONE segment of that episode (segment {k} of {n},
covering {t0:.1f}s-{t1:.1f}s of a {dur:.1f}s episode). Segment boundaries were detected from
fingertip force sensors, so a segment begins or ends where a hand grasped or released something.

Observed in this segment: {facts}

Reply with ONE short imperative sub-task label for what the robot is doing in this segment.
Use the style of a human annotator: lowercase, 2-6 words, no trailing period.
Examples: "reach for the deck", "grasp the card with the right hand",
"transfer the deck to the left hand", "insert the card into the slot".
Reply with the label ONLY — no explanation, no quotes."""


def spans_from_events(events_by_hand: dict, T: int, fps: float) -> list[tuple[int, int]]:
    """Merge both hands' contact transitions into ordered, hygiene-filtered spans."""
    min_f, max_f = int(MIN_SPAN_S * fps), int(MAX_SPAN_S * fps)
    cuts = sorted({int(f) for ev in events_by_hand.values() for f, _ in ev})

    # MIN_SPAN_S must hold between NEIGHBOURING cuts, not merely against the episode ends.
    # Filtering only against the ends let two events 3 frames apart (a right-hand release at 414
    # and a left-hand grasp at 416 -- the two halves of one handover) create a 0.1 s "sub-task",
    # and split single sub-tasks across two spans that the VLM then labelled identically.
    keep = []
    for c in cuts:
        if c < min_f or c > T - min_f:
            continue                      # too close to an episode edge
        if keep and c - keep[-1] < min_f:
            continue                      # too close to the previous accepted cut
        keep.append(c)

    bounds, out = [0] + keep + [T], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a <= max_f:
            out.append((a, b))
            continue
        n = int(np.ceil((b - a) / max_f))          # even subdivision of an over-long span
        edges = np.linspace(a, b, n + 1).astype(int)
        out += list(zip(edges[:-1], edges[1:]))
    return [(a, b) for a, b in out if b > a]


def span_facts(events_by_hand: dict, a: int, b: int, fps: float) -> str:
    bits = []
    for hand, ev in events_by_hand.items():
        inside = [(f, k) for f, k in ev if a <= f < b]
        if inside:
            bits.append(f"{hand} hand " + ", ".join(f"{k} at {(f - a) / fps:.1f}s" for f, k in inside))
    return "; ".join(bits) if bits else "no contact change (motion only)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="the dexlabel LeRobot view")
    ap.add_argument("--out", required=True, help="subtasks.json")
    ap.add_argument("--pipeline", required=True, help="code_labeling/pipeline root (for client/media)")
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--dry-run", action="store_true", help="split only; no VLM calls")
    ap.add_argument("--workers", type=int, default=8, help="concurrent episodes (IO-bound)")
    args = ap.parse_args()

    sys.path.insert(0, args.pipeline)
    from codegen.grounding.tactile import TACTILE_FINGERS, tactile_hand_events
    import pyarrow.parquet as papq

    if not args.dry_run:
        from codegen.actor import client, media
        key = client.load_key()

    D = args.dataset
    mod = json.load(open(f"{D}/meta/modality.json"))["state"]
    fps = float(json.load(open(f"{D}/meta/info.json")).get("fps", 30.0))
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in open(f"{D}/meta/tasks.jsonl")}
    eps = [json.loads(l) for l in open(f"{D}/meta/episodes.jsonl")]
    if args.episodes:
        eps = eps[: args.episodes]

    out, n_calls = {}, 0
    lock = __import__("threading").Lock()

    def do_episode(e):
        nonlocal n_calls
        i = e["episode_index"]
        st = np.stack(papq.read_table(f"{D}/data/chunk-000/episode_{i:06d}.parquet")
                      .column("observation.state").to_numpy(zero_copy_only=False))
        T = len(st)
        tac = {h: {f: st[:, mod[f"{h}_{f}_f6"]["start"]:mod[f"{h}_{f}_f6"]["end"]]
                   for f in TACTILE_FINGERS if f"{h}_{f}_f6" in mod}
               for h in ("left", "right")}
        ev, _ = tactile_hand_events(tac)
        spans = spans_from_events(ev, T, fps)
        instruction = e["tasks"][0] if e.get("tasks") else tasks.get(0, "")

        sub_id = np.zeros(T, dtype=np.int32)
        text = {}
        frames = None
        for k, (a, b) in enumerate(spans, start=1):
            sub_id[a:b] = k
            facts = span_facts(ev, a, b, fps)
            if args.dry_run:
                text[str(k)] = f"[span {k}] {facts}"
                continue
            if frames is None:
                import av
                c = av.open(f"{D}/videos/chunk-000/top/episode_{i:06d}.mp4")
                frames = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
                c.close()
            picks = np.linspace(a, min(b - 1, len(frames) - 1), FRAMES_PER_SPAN).astype(int)
            content = [media._txt(PROMPT.format(instruction=instruction, k=k, n=len(spans),
                                                t0=a / fps, t1=b / fps, dur=T / fps, facts=facts))]
            content += [media._img(media.frame_b64(media._to_pil(frames[p]))) for p in picks]
            # max_tokens must budget for REASONING, not just the answer: the actor model is a
            # thinking model, so a tight cap (32) is spent on hidden reasoning and the call dies
            # with finish_reason=length before emitting the 2-6 word label.
            label = client.call_model([{"role": "user", "content": content}], key,
                                      max_tokens=1024, temperature=0.1)
            label = label.strip().strip('"').strip().splitlines()[-1].strip().strip('"').lower()
            text[str(k)] = label
            with lock:
                n_calls += 1

        # MERGE adjacent spans the VLM gave the SAME label. A >MAX_SPAN_S subdivision, or two
        # contact events inside one sub-task, splits a single sub-task in two; the labeller
        # correctly names both identically. Left unmerged these become distinct sub_task ids, so
        # `sub_task_blocks` would emit two blocks where the episode has one action.
        if not args.dry_run and spans:
            m_spans, m_text, cur = [], {}, None
            for k, (a, b) in enumerate(spans, start=1):
                lab = text[str(k)]
                if cur is not None and lab == m_text[str(cur)]:
                    m_spans[-1] = (m_spans[-1][0], b)          # extend the previous block
                    continue
                cur = len(m_spans) + 1
                m_spans.append((a, b))
                m_text[str(cur)] = lab
            if len(m_spans) != len(spans):
                print(f"       merged {len(spans)} spans -> {len(m_spans)} sub-tasks", flush=True)
            spans, text = m_spans, m_text
            sub_id = np.zeros(T, dtype=np.int32)
            for k, (a, b) in enumerate(spans, start=1):
                sub_id[a:b] = k

        with lock:
            out[str(i)] = {"sub_id": sub_id.tolist(), "text": text, "fps": fps,
                           "instruction": instruction,
                           "spans": [[int(a), int(b)] for a, b in spans],
                           "events": {h: [[int(f), k] for f, k in v] for h, v in ev.items()}}
            print(f"  ep{i:03d} T={T} spans={len(spans)} "
                  + ("(dry)" if args.dry_run else "labels=" + " | ".join(list(text.values())[:3])),
                  flush=True)

    # Episode-level thread pool: every call is IO-bound on the router, so threads are the right
    # unit and the per-episode video decode stays off the critical path of the other workers.
    if args.workers > 1 and not args.dry_run:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for fut in [pool.submit(do_episode, e) for e in eps]:
                fut.result()
    else:
        for e in eps:
            do_episode(e)

    json.dump(out, open(args.out, "w"))
    print(f"[subtasks] {len(out)} episodes, {n_calls} VLM calls -> {args.out}", flush=True)
    print("SUBTASKS_DONE", flush=True)


if __name__ == "__main__":
    main()
