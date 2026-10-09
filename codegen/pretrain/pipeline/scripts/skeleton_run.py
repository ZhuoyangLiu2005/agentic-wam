#!/usr/bin/env python
"""Skeleton-cached labelling — the CLI for all three tiers.

    CODEGEN_CONFIG=configs/yam.yaml $PY scripts/skeleton_run.py <cmd> [...]

    inventory  --units <file> [--out outputs/<emb>/skeleton/inventory.json]
               Scan the work units (parquet only) -> which labels exist, and which
               episode should teach each one. No model calls, no video.

    generate   [--inventory F] [--limit N] [--coder-model M] [--parallel P]
               [--shard i/N]
               Stage A: fill the skeleton store. THE ONLY TIER THAT SPENDS CALLS.
               Resumable; ~15s per label and almost all of it network wait, so
               --parallel is worth it. Default 1: the router's rate limit is
               unknown, and stage A must never fan out by accident.
               Across PROCESSES pass disjoint --shard i/N — sharing a store is
               safe, but two unsharded workers label everything twice.

    assemble   --units <file> [--tag T] [--refine-budget N] [--parallel P]
               Stage B: build every episode's run dir from the cache. No model
               calls (unless --refine-budget > 0, which lets tier C regenerate at
               most N blocks per episode).

    stats      what is in the store, and what the inventory implies it will cost.

Run dirs are the same schema the per-episode pipeline writes, so `gate`,
`mine_label_patterns` and any cache builder work unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _store_root(emb: str) -> Path:
    from codegen import config
    p = config.get("SKELETON_STORE") or f"outputs/{emb}/skeleton/store"
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _units(path: str):
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        task, _, idx = line.partition(":")
        out.append((task.strip(), int(idx) if idx.strip() else 0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inventory", "generate", "assemble", "stats"])
    ap.add_argument("--units", default="")
    ap.add_argument("--inventory", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--limit-episodes", type=int, default=0)
    ap.add_argument("--refine-budget", type=int, default=0,
                    help="tier C: max blocks per episode to regenerate (0 = never)")
    ap.add_argument("--parallel", type=int, default=0,
                    help="workers (default: assemble 8, generate 1 — stage A's\n                          rate limit is unknown, so it does not fan out silently)")
    ap.add_argument("--shard", default="",
                    help="i/N — stage A partition for a MULTI-PROCESS fleet. "
                         "Two unsharded workers on one inventory generate "
                         "every label twice.")
    ap.add_argument("--coder-model", default="")
    ap.add_argument("--embodiment", default="")
    args = ap.parse_args()

    from codegen import config
    emb = args.embodiment or config.embodiment()
    import os
    os.environ["CODEGEN_EMBODIMENT"] = emb
    from codegen.actor.client import MODEL
    from codegen.skeleton import generate, inventory
    from codegen.skeleton.store import SkeletonStore

    coder_model = args.coder_model or MODEL
    inv_path = Path(args.inventory or args.out
                    or ROOT / "outputs" / emb / "skeleton" / "inventory.json")
    store = SkeletonStore(_store_root(emb))
    log = lambda m: print(m, file=sys.stderr, flush=True)

    if args.cmd == "inventory":
        if not args.units:
            log("inventory needs --units")
            return 2
        try:
            labels = inventory.build(_units(args.units), emb, log=log,
                                     limit_episodes=args.limit_episodes)
        except RuntimeError as e:
            log(f"[inventory] FAILED ({emb}): {e}")
            return 1
        if not labels:
            # never overwrite a real inventory with an empty one, and never
            # report an empty scan as success
            log(f"[inventory] no labels found ({emb}) — refusing to write "
                f"an empty inventory to {inv_path}")
            return 1
        inventory.save(labels, inv_path)
        n_inst = sum(r["n"] for r in labels.values())
        modal = sum(r["modal_share"] * r["n"] for r in labels.values()) / max(n_inst, 1)
        print(json.dumps({"embodiment": emb, "labels": len(labels),
                          "block_instances": n_inst,
                          "amortisation": round(n_inst / max(len(labels), 1), 1),
                          "modal_signature_share": round(modal, 3),
                          "inventory": str(inv_path)}, indent=1))
        return 0

    if args.cmd == "generate":
        if not inv_path.exists():
            log(f"no inventory at {inv_path} — run `inventory` first")
            return 2
        labels = inventory.load(inv_path)
        summary = generate.run(labels, store, coder_model, limit=args.limit,
                               log=log, parallel=args.parallel or 1,
                               shard=args.shard)
        print(json.dumps(summary, indent=1))
        return 0

    if args.cmd == "stats":
        pv = generate.prompt_version(coder_model)
        s = store.stats()
        s["prompt_version"] = pv
        s["store"] = str(store.root)
        if inv_path.exists():
            labels = inventory.load(inv_path)
            from codegen.skeleton.store import skeleton_key
            have = sum(1 for lab in labels if store.get(skeleton_key(lab, pv, emb)))
            n_inst = sum(r["n"] for r in labels.values())
            s.update(labels=len(labels), have=have, missing=len(labels) - have,
                     block_instances=n_inst,
                     amortisation=round(n_inst / max(len(labels), 1), 1))
        print(json.dumps(s, indent=1))
        return 0

    # ---- assemble -----------------------------------------------------------
    if not args.units:
        log("assemble needs --units")
        return 2
    tag = args.tag or config.get("TAG")
    if not tag:
        log("no tag: set `tag:` in the config or pass --tag")
        return 1
    from codegen.skeleton.assemble import assemble
    pv = generate.prompt_version(coder_model)
    units = _units(args.units)
    outdir = ROOT / "outputs" / emb / f"skeleton_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"[assemble] {len(units)} units, prompt_version={pv}, store={store.root} "
        f"({len(store)} skeletons), refine_budget={args.refine_budget}")
    t0 = time.time()
    rows = []
    mf = (outdir / "manifest.jsonl").open("a")

    def one(u):
        try:
            return assemble(u[0], u[1], store, pv, tag, coder_model,
                            refine_budget=args.refine_budget, log=log)
        except Exception as e:
            return {"task": u[0], "episode": f"episode_{u[1]:03d}",
                    "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=max(args.parallel or 8, 1)) as ex:
        futs = {ex.submit(one, u): u for u in units}
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            mf.write(json.dumps(r) + "\n")
            mf.flush()
            if n % 25 == 0 or n == len(units):
                log(f"[assemble] {n}/{len(units)}")
    mf.close()
    ok = [r for r in rows if not r.get("error")]
    sl = sum(r.get("slices", 0) for r in ok)
    summary = {
        "tag": tag, "units": len(rows), "errored": len(rows) - len(ok),
        "slices": sl,
        "matched_slice_rate": round(sum(r.get("matched", 0) for r in ok) / max(sl, 1), 3),
        # matched BEFORE repair — the number that cannot be gamed by emitting
        # no gripper call and letting repair_tail copy the detector in
        "matched_raw_rate": round(sum(r.get("matched_raw", 0) for r in ok) / max(sl, 1), 3),
        "tail_repaired_slices": sum(r.get("tail_repaired", 0) for r in ok),
        "over_emission_linted_slices": sum(r.get("over_emission_linted", 0) for r in ok),
        "regenerated_slices": sum(r.get("regenerated", 0) for r in ok),
        "missing_skeleton_slices": sum(r.get("missing_skeleton", 0) for r in ok),
        "grip_pins": sum(r.get("grip_pins", 0) for r in ok),
        "observed_events": sum(r.get("observed_events", 0) for r in ok),
        "wall_seconds": round(time.time() - t0, 1),
        "seconds_per_episode": round((time.time() - t0) / max(len(rows), 1), 3),
    }
    summary["embodiment"] = emb
    summary["pin_rate"] = round(summary["grip_pins"] / max(summary["observed_events"], 1), 3)
    # matched/pin are BOTH gameable: a skeleton that emits no gripper call at all
    # lets repair_tail transcribe every observed event from the detector and
    # scores 1.0/1.0 with zero model-written contact code. Report the share of
    # slices that needed repair next to them so the headline cannot stand alone.
    summary["repaired_share"] = round(
        (summary["tail_repaired_slices"] + summary["over_emission_linted_slices"])
        / max(sl, 1), 3)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    if rows and len(ok) == 0:
        log(f"[assemble] every one of {len(rows)} unit(s) errored — "
            f"first: {next((r.get('error') for r in rows if r.get('error')), '?')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
