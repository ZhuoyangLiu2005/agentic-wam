"""Gate: every episode's SERVED instruction (parquet task_index -> tasks.jsonl) must match
meta/episodes.jsonl AND have a precomputed Cosmos text embed.

Exists because a launch gate of `assert n_embeds > 0` let a run train for 1.3k steps with every
episode conditioned on the first task string (task_index was never rewritten by the relabeller).
`scripts/preflight_sft.py` runs `check()` on the shards a data config names.
"""
import argparse, collections, glob, hashlib, json, os, sys
import pyarrow.parquet as pq

TPL = "A video recorded from a robot's point of view executing the following instruction: {task}"


def check(shards, cache_dir):
    """-> (census {instruction: n_episodes}, [(parquet, why)] mismatches, [instruction] missing embeds)."""
    have = {f.split(".")[0] for f in os.listdir(cache_dir)}
    cen, bad, miss = collections.Counter(), [], []
    for sh in shards:
        tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(sh + "/meta/tasks.jsonl")}
        eptask = {int(json.loads(l)["episode_index"]): json.loads(l)["tasks"][0] for l in open(sh + "/meta/episodes.jsonl")}
        for f in sorted(glob.glob(sh + "/data/chunk-*/*.parquet")):
            t = pq.read_table(f, columns=["episode_index", "task_index"])
            ep = int(t.column("episode_index").to_pylist()[0])
            tis = set(t.column("task_index").to_pylist())
            if len(tis) != 1:
                bad.append((f, "mixed task_index"))
                continue
            served = tasks[tis.pop()]
            if served != eptask[ep]:
                bad.append((f, f"served={served!r} meta={eptask[ep]!r}"))
            cen[served] += 1
            if hashlib.sha256(TPL.format(task=served).encode("utf-8")).hexdigest() not in have:
                miss.append(served)
    return cen, bad, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="a directory of *shard_* LeRobot shards")
    ap.add_argument("--cache", default=None, help="default: <dataset>/cosmos_text_embeds_cache")
    a = ap.parse_args()
    cache_dir = a.cache or os.path.join(a.dataset, "cosmos_text_embeds_cache")
    cen, bad, miss = check(sorted(glob.glob(os.path.join(a.dataset, "*shard_*"))), cache_dir)
    print(json.dumps({"episodes": sum(cen.values()), "census": cen, "mismatch": len(bad), "embed_miss": len(set(miss))}))
    for f, why in bad[:10]:
        print("MISMATCH", f, why, file=sys.stderr)
    for m in sorted(set(miss)):
        print("MISSING EMBED", repr(m), file=sys.stderr)
    sys.exit(1 if (bad or miss) else 0)


if __name__ == "__main__":
    main()
