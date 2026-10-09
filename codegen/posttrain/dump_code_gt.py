"""Dump a policy-code GT cache to plain text so a human can read what the model is supervised on.

The caches are torch files ({"codes": [str], "code_id_by_idx": LongTensor[n_frames]}), which is
convenient for training and useless for inspection. This writes one readable file: per episode, the
frame range of every code segment followed by the code body, plus a header with coverage stats.
"""
import argparse, glob, json, os
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache", required=True, help="basename of the policy_code_idx_*.pt in --dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    c = torch.load(os.path.join(a.dataset, a.cache), map_location="cpu")
    codes, ids = c["codes"], c["code_id_by_idx"]
    shards = sorted(glob.glob(os.path.join(a.dataset, "train_shard_*")),
                    key=lambda p: int(p.rsplit("_", 1)[1]))
    eps = []
    for sh in shards:
        tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(sh + "/meta/tasks.jsonl")}
        for l in open(sh + "/meta/episodes.jsonl"):
            d = json.loads(l)
            eps.append((os.path.basename(sh), d["episode_index"], d["length"], d["tasks"][0]))

    total = sum(e[2] for e in eps)
    with open(a.out, "w") as f:
        f.write(f"# code GT dump: {a.dataset}\n# cache: {a.cache}\n")
        f.write(f"# {len(eps)} episodes, {total} frames, {len(codes)-1} distinct codes, "
                f"{int((ids==0).sum())} frames with NO code\n\n")
        off = 0
        for i, (sh, ep, L, instr) in enumerate(eps):
            if a.max_episodes and i >= a.max_episodes:
                break
            seg = ids[off:off + L]
            f.write("=" * 100 + f"\n{sh} episode {ep}  ({L} frames)\nINSTRUCTION: {instr}\n" + "=" * 100 + "\n")
            s = 0
            for t in range(1, L + 1):
                if t == L or int(seg[t]) != int(seg[s]):
                    cid = int(seg[s])
                    f.write(f"\n--- frames {off+s}-{off+t-1} (local {s}-{t-1}, {t-s} frames)  code_id={cid}\n")
                    f.write((codes[cid] if cid else "<NO CODE>").rstrip() + "\n")
                    s = t
            off += L
            f.write("\n")
    print(json.dumps({"out": a.out, "episodes": len(eps), "frames": total,
                      "bytes": os.path.getsize(a.out)}))


if __name__ == "__main__":
    main()
