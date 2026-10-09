"""Bridge pipeline A's outputs into the FLAT per-frame code cache the trainer consumes.

`build_flat_code_cache.py` was written for the ASPIRE layout
    <aspire_root>/<trajectory_id>/vlm_s2_subtask_fused_<tag>/{program.py, episode_000_lines.json, trace.json}
while pipeline A writes
    outputs/<emb>/<task>/<model>/episode_NNN/<tag>/{program.py, lines.json, trace.json}
Rather than fork the builder (it owns the dedup + the alignment contract, both easy to get subtly
wrong), this materialises a thin view in the expected shape -- symlinks, not copies -- and hands it
to the real builder.

THE ALIGNMENT CONTRACT is the part that matters. `code_id_by_idx` is indexed by GLOBAL frame number
in the TRAINING dataset's enumeration order, so `episode_order.json` must list episodes in exactly
the order `RobotVideoDataset` walks them: shard by shard in `dataset_dirs` order, episode by episode
within each shard, with `offset` the running frame total. It is built here from the TRAINING view,
never from the labelling view -- the two hold the same episodes and the same
per-episode lengths (verified: identical grids), but only the training view defines the order the
loader will use. `verify_code_alignment.py` re-checks this afterwards; do not skip it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--embodiment", required=True, choices=["yam_sft", "tianji_sharpa"],
                    help="the engine embodiment that labeled the episodes (outputs/<embodiment>/...)")
    ap.add_argument("--train-view", required=True, help="the TRAINING LeRobot view (frame order)")
    ap.add_argument("--task", required=True, help="pipeline task name (labeling view)")
    ap.add_argument("--tag", required=True, help="labeling run tag")
    ap.add_argument("--model", default="gemini-3.1-pro")
    ap.add_argument("--work", required=True, help="scratch dir for the adapter view")
    ap.add_argument("--out", required=True, help="policy_code_idx_*.pt")
    a = ap.parse_args()

    src_root = os.path.join(a.pipeline, "outputs", a.embodiment, a.task, a.model)

    # ---- episode_order.json, in TRAINING-dataset enumeration order --------------------------
    shards = sorted(glob.glob(os.path.join(a.train_view, "train_shard_*")),
                    key=lambda p: int(p.rsplit("_", 1)[1]))
    order, offset, ep_global = [], 0, 0
    for sh in shards:
        eps = [json.loads(l) for l in open(os.path.join(sh, "meta/episodes.jsonl"))]
        for e in sorted(eps, key=lambda x: x["episode_index"]):
            # `subset` is the builder's DIRECTORY KEY -- it looks up
            #     <aspire_root>/<subset>/vlm_s2_subtask_fused_<tag>/
            # so it must be the per-episode dir name, not the shard. (ABC's dirs are named
            # "<task>__ep<NNNNNN>" for exactly this reason.) The shard identity is kept in
            # trajectory_id, which the builder only passes through.
            order.append({"trajectory_id": f"{os.path.basename(sh)}:{e['episode_index']}",
                          "subset": f"{a.task}__ep{ep_global:06d}",
                          "offset": offset, "length": int(e["length"])})
            offset += int(e["length"])
            ep_global += 1
    order_path = os.path.join(a.work, "episode_order.json")
    os.makedirs(a.work, exist_ok=True)
    json.dump(order, open(order_path, "w"), indent=1)
    print(f"[order] {len(order)} episodes, {offset} frames -> {order_path}")

    # ---- adapter view: symlink pipeline A artifacts into the ASPIRE shape -------------------
    linked = missing = 0
    for i, ep in enumerate(order):
        src = os.path.join(src_root, f"episode_{i:03d}", a.tag)
        dst = os.path.join(a.work, "aspire", ep["subset"], f"vlm_s2_subtask_fused_{a.tag}")
        os.makedirs(dst, exist_ok=True)
        prog, lines = os.path.join(src, "program.py"), os.path.join(src, "lines.json")
        if not (os.path.exists(prog) and os.path.exists(lines)):
            missing += 1
            continue
        for s, d in ((prog, "program.py"), (lines, "episode_000_lines.json"),
                     (os.path.join(src, "trace.json"), "trace.json")):
            t = os.path.join(dst, d)
            if os.path.lexists(t):
                os.unlink(t)
            if os.path.exists(s):
                os.symlink(s, t)
        linked += 1
    print(f"[adapt] linked {linked}, missing {missing}")
    if missing:
        print("        (missing episodes become empty code, exactly as the builder handles a gap)")

    # ---- the real builder -------------------------------------------------------------------
    bd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_flat_code_cache.py")
    cmd = [sys.executable, bd, "--aspire-root", os.path.join(a.work, "aspire"),
           "--tag", a.tag, "--order", order_path, "--out", a.out]
    print("[build]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:] or r.stderr[-3000:])
    if r.returncode != 0:
        sys.exit(f"build_code_cache failed rc={r.returncode}")
    print("TRAINABLE_CACHE_DONE")


if __name__ == "__main__":
    main()
