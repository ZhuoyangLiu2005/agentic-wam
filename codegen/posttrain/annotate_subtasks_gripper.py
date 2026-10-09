"""Sub-task segmentation for a gripper robot: GRIPPER-TRANSITION SPLITS, then a VLM names the spans.

Writes <shard>/subtasks.json, which the code_labeling pipeline reads INSTEAD of a human
annotation.sub_task column. Use it for a dataset that has no human sub-task annotation.

Same design as tianji's annotate_subtasks_tactile.py -- a direct sensor decides WHERE sub-tasks
change, the VLM only answers WHAT is happening -- with the event source swapped: YAM has no
tactile, but pipeline A's own `grounding/transitions.py` (velocity-sign gripper detector +
hover-chatter debounce, THE detector the yam pretrain labeling used) gives frame-accurate
open/close events from the analog gripper channel (dims 32/33, ~1=open).

All the tianji lessons are baked in: +1-frame boundary shift (an event terminates the span that
caused it), min-span enforced between NEIGHBOURING cuts, adjacent same-label spans merged after
naming, generous max_tokens for the thinking actor model.

Output: per-shard `<shard>/subtasks.json` keyed by local episode index:
    {"sub_id": [T], "text": {id: label}, "spans": [[a,b]...], "instruction": str, "events": {...}}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading

import numpy as np

MIN_SPAN_S = 1.0
MAX_SPAN_S = 8.0
FRAMES_PER_SPAN = 4
FPS = 30.0

PROMPT = """You are labelling one segment of a bimanual robot manipulation episode.

Overall task: {instruction}

The frames below are consecutive moments from ONE segment (segment {k} of {n}, {t0:.1f}s-{t1:.1f}s
of a {dur:.1f}s episode). Segment boundaries were detected from the gripper sensors, so a segment
begins or ends where a gripper closed on or released something.

Observed in this segment: {facts}

Reply with ONE short imperative sub-task label: lowercase, 2-6 words, no trailing period.
Examples: "reach for a ball", "grasp a ball with the right gripper", "drop the ball into the box",
"pull the zipper up", "fold the left sleeve inward".
Reply with the label ONLY -- no explanation, no quotes."""


def spans_from_events(events_by_arm, T, fps):
    min_f, max_f = int(MIN_SPAN_S * fps), int(MAX_SPAN_S * fps)
    cuts = sorted({int(f) + 1 for ev in events_by_arm.values() for f, _ in ev})  # +1: event ends its span
    keep = []
    for c in cuts:
        if c < min_f or c > T - min_f:
            continue
        if keep and c - keep[-1] < min_f:
            continue
        keep.append(c)
    bounds, out = [0] + keep + [T], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a <= max_f:
            out.append((a, b)); continue
        n = int(np.ceil((b - a) / max_f))
        e = np.linspace(a, b, n + 1).astype(int)
        out += list(zip(e[:-1], e[1:]))
    return [(a, b) for a, b in out if b > a]


def span_facts(events_by_arm, a, b, fps):
    bits = []
    for arm, ev in events_by_arm.items():
        inside = [(f, k) for f, k in ev if a <= f + 1 < b or a <= f < b]
        if inside:
            bits.append(f"{arm} gripper " + ", ".join(f"{k} at {(f - a) / fps:.1f}s" for f, k in inside))
    return "; ".join(bits) if bits else "no gripper change (motion only)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--shards", default="", help="comma list; empty = all")
    ap.add_argument("--episodes", type=int, default=0, help="per-shard cap, 0=all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.pipeline)
    os.environ.setdefault("CODEGEN_EMBODIMENT", "yam_sft")
    import pandas as pd
    from codegen.grounding.transitions import debounce_events, grip_transitions
    if not args.dry_run:
        from codegen.actor import client, media
        key = client.load_key()
    lock = threading.Lock()
    n_calls = 0

    shards = args.shards.split(",") if args.shards else sorted(
        (p for p in os.listdir(args.dataset) if p.startswith("train_shard_")),
        key=lambda n: int(n.rsplit("_", 1)[1]))

    def do_shard(shard):
        nonlocal n_calls
        d = os.path.join(args.dataset, shard)
        eps = [json.loads(l) for l in open(os.path.join(d, "meta/episodes.jsonl"))]
        if args.episodes:
            eps = eps[: args.episodes]
        out_p = os.path.join(d, "subtasks.json")
        out = json.loads(open(out_p).read()) if os.path.exists(out_p) else {}
        for e in eps:
            i = e["episode_index"]
            if str(i) in out and not args.dry_run:
                continue                                    # resumable
            df = pd.read_parquet(os.path.join(d, f"data/chunk-000/episode_{i:06d}.parquet"))
            st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
            T = len(st)
            ev = {}
            for arm, g in (("left", 32), ("right", 33)):
                pos = st[:, (0 if arm == "left" else 16):][:, :16].reshape(T, 4, 4)[:, :3, 3]
                raw = grip_transitions(st[:, g])
                ev[arm], _ = debounce_events(raw, pos)
            spans = spans_from_events(ev, T, FPS)
            instruction = e["tasks"][0]
            frames = None
            sub_id = np.zeros(T, dtype=np.int32); text = {}
            for k, (a, b) in enumerate(spans, start=1):
                sub_id[a:b] = k
                facts = span_facts(ev, a, b, FPS)
                if args.dry_run:
                    text[str(k)] = f"[span {k}] {facts}"; continue
                if frames is None:
                    import av
                    c = av.open(os.path.join(d, f"videos/chunk-000/observation.images.cam_high/episode_{i:06d}.mp4"))
                    frames = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)]); c.close()
                picks = np.linspace(a, min(b - 1, len(frames) - 1), FRAMES_PER_SPAN).astype(int)
                content = [media._txt(PROMPT.format(instruction=instruction, k=k, n=len(spans),
                                                    t0=a / FPS, t1=b / FPS, dur=T / FPS, facts=facts))]
                content += [media._img(media.frame_b64(media._to_pil(frames[p]))) for p in picks]
                lab = client.call_model([{"role": "user", "content": content}], key,
                                        max_tokens=1024, temperature=0.1)
                lab = lab.strip().strip('"').splitlines()[-1].strip().strip('"').lower()
                text[str(k)] = lab
                with lock:
                    n_calls += 1
            if not args.dry_run and spans:                  # merge adjacent identical labels
                m_spans, m_text, cur = [], {}, None
                for k, (a, b) in enumerate(spans, start=1):
                    lab = text[str(k)]
                    if cur is not None and lab == m_text[str(cur)]:
                        m_spans[-1] = (m_spans[-1][0], b); continue
                    cur = len(m_spans) + 1; m_spans.append((a, b)); m_text[str(cur)] = lab
                spans, text = m_spans, m_text
                sub_id = np.zeros(T, dtype=np.int32)
                for k, (a, b) in enumerate(spans, start=1):
                    sub_id[a:b] = k
            with lock:
                out[str(i)] = {"sub_id": sub_id.tolist(), "text": text, "instruction": instruction,
                               "spans": [[int(a), int(b)] for a, b in spans],
                               "events": {h: [[int(f), k] for f, k in v] for h, v in ev.items()}}
                print(f"  {shard}/ep{i:02d} T={T} spans={len(spans)} "
                      + ("(dry)" if args.dry_run else "| " + " | ".join(list(text.values())[:3])), flush=True)
        json.dump(out, open(out_p, "w"))

    if args.workers > 1 and not args.dry_run:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for f in [pool.submit(do_shard, s) for s in shards]:
                f.result()
    else:
        for s in shards:
            do_shard(s)
    print(f"[subtasks] shards={len(shards)} VLM calls={n_calls}", flush=True)
    print("SUBTASKS_DONE", flush=True)


if __name__ == "__main__":
    main()
