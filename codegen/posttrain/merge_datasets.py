"""Concatenate LeRobot shard-datasets into one, keeping the code-GT caches aligned.

The training cache is a FLAT per-frame array laid out in shard order, so a merge is only valid if the
shards are appended whole, in order, and the caches are concatenated in exactly that order. This script
copies each source's shards in sequence (renumbering), concatenates the caches, rebuilds episode_order,
and then re-verifies the total frame count against the merged meta before anything is written as final.
"""
from __future__ import annotations

import argparse, glob, json, os, shutil

import torch


def shards(d):
    return sorted(glob.glob(os.path.join(d, "train_shard_*")), key=lambda p: int(p.rsplit("_", 1)[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="source datasets, in the order they should appear")
    ap.add_argument("--cache-name", required=True, help="code-GT cache file name present in every source")
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    k = 0
    codes, idx_of, flat, order, offset, manifest = [""], {"": 0}, [], [], 0, []
    for src in args.src:
        c = torch.load(os.path.join(src, args.cache_name))
        src_codes, src_ids = c["codes"], c["code_id_by_idx"]
        man = {(m["shard"], m["episode_index"]): m for m in json.load(open(os.path.join(src, "source_manifest.json")))}
        pos = 0
        for sh in shards(src):
            si = int(sh.rsplit("_", 1)[1])
            out = os.path.join(args.dst, f"train_shard_{k}")
            if not os.path.exists(out):
                shutil.copytree(sh, out)
            for e in [json.loads(l) for l in open(os.path.join(sh, "meta/episodes.jsonl"))]:
                n = int(e["length"])
                ids = src_ids[pos:pos + n]
                # re-index this episode's codes into the merged table
                remap = {}
                for v in ids.unique().tolist():
                    s = src_codes[v]
                    if s not in idx_of:
                        idx_of[s] = len(codes); codes.append(s)
                    remap[v] = idx_of[s]
                flat.append(torch.tensor([remap[int(v)] for v in ids], dtype=torch.int32))
                m = man[(si, e["episode_index"])]
                manifest.append({"shard": k, "episode_index": e["episode_index"], "source": m["source"],
                                 "length": n, "max_skew_s": m.get("max_skew_s")})
                order.append({"trajectory_id": f"{k}:{e['episode_index']}", "offset": offset, "length": n})
                offset += n; pos += n
            k += 1
        assert pos == int(src_ids.numel()), f"{src}: consumed {pos} of {int(src_ids.numel())} cache frames"
    flat_t = torch.cat(flat)
    total = sum(json.loads(l)["length"] for sh in shards(args.dst) for l in open(os.path.join(sh, "meta/episodes.jsonl")))
    assert int(flat_t.numel()) == total == offset, (int(flat_t.numel()), total, offset)
    assert int((flat_t == 0).sum()) == 0, "empty slices in the merged cache"
    torch.save({"codes": codes, "code_id_by_idx": flat_t}, os.path.join(args.dst, args.cache_name))
    json.dump(order, open(os.path.join(args.dst, "episode_order.json"), "w"))
    json.dump(manifest, open(os.path.join(args.dst, "source_manifest.json"), "w"), indent=1)
    print(f"[merge] {k} shards, {len(order)} episodes, {total} frames, {len(codes)-1} codes, 0 empty -> {args.dst}")


if __name__ == "__main__":
    main()
