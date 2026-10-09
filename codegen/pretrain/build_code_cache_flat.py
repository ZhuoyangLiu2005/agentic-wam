#!/usr/bin/env python3
"""Engine outputs of one pretraining source -> its FLAT code cache {codes, code_id_by_idx}.

    python codegen/pretrain/build_code_cache_flat.py --emb yam|ego --dataset-root <dir> --out <cache.pt>

`code_id_by_idx` is indexed by GLOBAL frame number in the order the training loader enumerates the
source (docs/CODE_ANNOTATION.md §3): the `train_shard_*` datasets under --dataset-root in numeric
order, episodes by index within a shard, frames in order. That order is written next to the cache
as `episode_order.json` (check it with codegen/posttrain/verify_code_alignment.py).

Per frame the target is that frame's CODE SLICE: the Stage-3 run dir of the episode
(`<outputs>/<emb>/<dataset>/<model>/episode_NNN/<tag>`) holds `program.py` and `lines.json`, whose
`frame_src_block[f] = [a, b]` (1-indexed, inclusive) selects program lines a..b. An episode with no
run dir gets the empty code "" (masked out of the loss, not trained as empty).
"""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

DEFAULT_OUTPUTS = Path(__file__).resolve().parent / "pipeline" / "outputs"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def episode_order(dataset_root):
    """[{trajectory_id, subset, episode_index, offset, length}] in the loader's frame order."""
    shards = sorted((p for p in Path(dataset_root).glob("train_shard_*") if (p / "meta" / "info.json").exists()),
                    key=lambda p: int(p.name.rsplit("_", 1)[1]))
    if not shards:
        raise SystemExit(f"no train_shard_*/meta/info.json under {dataset_root}")
    order, offset = [], 0
    for sh in shards:
        eps = [json.loads(l) for l in open(sh / "meta" / "episodes.jsonl") if l.strip()]
        for e in sorted(eps, key=lambda x: x["episode_index"]):
            L = int(e["length"])
            order.append({"trajectory_id": f"{sh.name}:{e['episode_index']}", "subset": sh.name,
                          "episode_index": int(e["episode_index"]), "offset": offset, "length": L})
            offset += L
    return order, offset


def _dir(outputs, model, tag, emb, key, ep):
    base = f"{outputs}/{emb}/{key}/{model}"
    for name in (f"episode_{ep:03d}", f"episode_{ep}"):
        d = f"{base}/{name}/{tag}"
        if os.path.isdir(d):
            return d
    return None


def one(job):
    """-> (offset, length, uniq_blocks, per_frame_block_id) or (offset, length, None, None)."""
    (outputs, model, tag), emb, key, ep, off, L = job
    d = _dir(outputs, model, tag, emb, key, ep)
    if d is None:
        return off, L, None, None
    try:
        prog = open(f"{d}/program.py").read().splitlines()
        fsb = json.load(open(f"{d}/lines.json"))["frame_src_block"]
    except Exception:                                                  # noqa: BLE001
        return off, L, None, None
    uniq, ids, seen = [], np.zeros(L, np.int32), {}
    for f in range(L):
        a, b = fsb[f] if f < len(fsb) else fsb[-1]
        blk = "\n".join(prog[max(a - 1, 0):b]).strip()
        j = seen.get(blk)
        if j is None:
            j = len(uniq); seen[blk] = j; uniq.append(blk)
        ids[f] = j
    return off, L, uniq, ids


def build(emb, dataset_root, out_path, outputs, model, tag, workers, limit):
    order, n_frames = episode_order(dataset_root)
    if limit:                             # smoke runs: pass a scratch --out, never the real cache
        order = order[:limit]
        n_frames = order[-1]["offset"] + order[-1]["length"]
    order_path = Path(out_path).with_name("episode_order.json")
    json.dump(order, open(order_path, "w"), indent=1)
    log(f"{emb}: {len(order):,} episodes, {n_frames:,} frames -> {out_path} (order: {order_path})")

    src = (str(outputs), model, tag)
    jobs = [(src, emb, e["subset"], e["episode_index"], e["offset"], e["length"]) for e in order]
    code_id = np.zeros(n_frames, np.int32)
    codes, code_of = [""], {"": 0}
    done = missing = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for off, L, uniq, ids in ex.map(one, jobs, chunksize=64):
            if uniq is None:
                missing += 1; continue
            remap = np.empty(len(uniq), np.int32)
            for j, blk in enumerate(uniq):
                g = code_of.get(blk)
                if g is None:
                    g = len(codes); code_of[blk] = g; codes.append(blk)
                remap[j] = g
            code_id[off:off + L] = remap[ids]
            done += 1
            if done % 20000 == 0:
                log(f"  {done:,}/{len(jobs):,} eps  codes={len(codes):,}  missing={missing}  {time.time()-t0:.0f}s")
    nz = int((code_id != 0).sum())
    log(f"{emb}: episodes ok={done:,} missing={missing}  codes={len(codes):,}  "
        f"labelled={nz/max(n_frames, 1):.4f}  {time.time()-t0:.0f}s")
    torch.save({"codes": codes, "code_id_by_idx": torch.from_numpy(code_id)}, out_path)
    log(f"wrote {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True, choices=["yam", "ego"])
    ap.add_argument("--dataset-root", required=True, help="the source's dataset directory (train_shard_*/)")
    ap.add_argument("--out", required=True, help="policy_code_idx_<name>.pt (episode_order.json is written beside it)")
    ap.add_argument("--outputs", default=str(DEFAULT_OUTPUTS), help="the engine's outputs/ root")
    ap.add_argument("--model", default="gemini-3.1-pro", help="actor model directory name")
    ap.add_argument("--tag", required=True, help="the campaign tag the Stage-3 `assemble` run wrote under")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="first N episodes only (smoke)")
    a = ap.parse_args()
    build(a.emb, a.dataset_root, a.out, a.outputs, a.model, a.tag, a.workers, a.limit)
