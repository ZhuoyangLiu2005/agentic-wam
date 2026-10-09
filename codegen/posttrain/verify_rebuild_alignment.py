"""Check a rebuilt dataset against the one a code-GT cache was built for, episode by episode.

The cache is a flat per-frame array concatenated in shard order, so it only transfers if the rebuild has
the SAME episodes in the SAME order with the SAME lengths. Anything else (an episode dropped by the
aligner, a different subset, a reordering) silently shifts every label after the first difference, which
is exactly the kind of bug that is invisible in the loss curve.
"""
from __future__ import annotations

import argparse, glob, json, os
import torch


def episodes(dst):
    man = {(m["shard"], m["episode_index"]): m["source"] for m in
           json.load(open(os.path.join(dst, "source_manifest.json")))}
    out = []
    for sh in sorted(glob.glob(os.path.join(dst, "train_shard_*")), key=lambda p: int(p.rsplit("_", 1)[1])):
        k = int(sh.rsplit("_", 1)[1])
        for e in [json.loads(l) for l in open(os.path.join(sh, "meta/episodes.jsonl"))]:
            out.append((man[(k, e["episode_index"])], int(e["length"]), e["tasks"][0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--cache", default="", help="code-GT cache whose frame count must match the new dataset")
    args = ap.parse_args()
    A, B = episodes(args.old), episodes(args.new)
    print(f"[align] old {len(A)} eps / {sum(a[1] for a in A)} frames | new {len(B)} eps / {sum(b[1] for b in B)} frames")
    bad = 0
    for i, (a, b) in enumerate(zip(A, B)):
        if a[0] != b[0] or a[1] != b[1] or a[2] != b[2]:
            bad += 1
            if bad <= 5:
                print(f"  MISMATCH {i}: old {os.path.basename(a[0])} len={a[1]} | new {os.path.basename(b[0])} len={b[1]}")
    ok = (len(A) == len(B)) and bad == 0
    if args.cache:
        n = int(torch.load(args.cache)["code_id_by_idx"].numel())
        print(f"[align] cache frames {n} vs new {sum(b[1] for b in B)} -> {'OK' if n == sum(b[1] for b in B) else 'MISMATCH'}")
        ok = ok and n == sum(b[1] for b in B)
    print("ALIGN_OK" if ok else "ALIGN_FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
