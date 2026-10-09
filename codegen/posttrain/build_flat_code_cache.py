#!/usr/bin/env python3
"""BRIDGE: ASPIRE inverse-code-gen outputs -> the flat per-frame code cache.

Each ASPIRE episode dir (data/yam/<task>/vlm_s2_subtask_fused_<tag>/) holds `program.py` +
`episode_000_lines.json` with `frame_src_block[t] = [first_line, last_line]` (1-indexed, inclusive)
= the sub-task code block active at frame t. This assembles the flat cache the loader consumes:

    policy_code_idx.pt = {codes: [unique block strings], code_id_by_idx: int32[num_frames]}
    episode_order.json = [{trajectory_id, subset, offset, length}]  (the alignment contract)

per-frame code = codes[code_id_by_idx[global_frame]]. The dataset's frame enumeration MUST equal
episode_order (checked by verify_code_alignment.py). Mirrors the (lost) July-5 build_code_cache + the
obs-idx bridge (build_obs_idx_cache.py; removed)."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np


def block_string(program_lines: list[str], first_last) -> str:
    """The sub-task code block [first_line, last_line] (1-indexed, inclusive) from program.py."""
    if not first_last:
        return ""
    a, b = int(first_last[0]), int(first_last[1])
    return "\n".join(program_lines[a - 1:b]).strip() + "\n"


def episode_frame_codes(program_py: Path, lines_json: Path) -> list[str]:
    """Per-frame code block string for one episode (len == episode length)."""
    prog = program_py.read_text().splitlines()
    fsb = json.loads(lines_json.read_text())["frame_src_block"]
    return [block_string(prog, blk) for blk in fsb]


def _canonical(block, strip_comments):
    """H3 comment-normalized code for dedup: drop comment-only lines (sub-task headers + coder
    rationale) and blanks. SAFE (line-based, never touches string contents) -> collapses blocks that
    differ only in prose to one target, ~halving the vocab. strip_comments=False returns verbatim."""
    if not strip_comments:
        return block
    kept = [ln for ln in block.split("\n") if ln.strip() and not ln.strip().startswith("#")]
    code = re.sub(r"-?\d+\.\d+", "0.0", "\n".join(kept))   # zero pose-delta FLOATS (motion detail the
    # action expert owns); keep int args like turns=/index= (semantic). Collapses blocks differing only in
    # IK coordinates -> the code target is the semantic plan, not memorized floats.
    return (code + "\n") if code.strip() else ""


def assemble_flat_cache(per_episode_frame_codes, strip_comments=False):
    """episodes' per-frame block strings -> {codes(unique), code_id_by_idx(int32)}. Dedups by the
    (optionally comment-normalized) string."""
    codes: list[str] = []
    index: dict[str, int] = {}
    ids: list[int] = []
    EMPTY = "def policy_chunk(obs, state, robot, skills):\n    return skills.visual_diff(obs, obs)\n"
    for frames in per_episode_frame_codes:
        for c in frames:
            c = _canonical(c or EMPTY, strip_comments) or EMPTY
            if c not in index:
                index[c] = len(codes); codes.append(c)
            ids.append(index[c])
    return codes, np.asarray(ids, dtype=np.int32)


def main() -> int:
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aspire-root", required=True, help="dir containing <task>/vlm_s2_subtask_fused_<tag>/")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--order", required=True, help="episode_order.json: [{trajectory_id,subset,offset,length}] in DATASET order")
    ap.add_argument("--out", required=True, help="output policy_code_idx.pt")
    ap.add_argument("--require-skills-key", default="",
                    help="only consume runs whose trace.json skills_key matches (freeze-consistency, B4)")
    ap.add_argument("--strip-comments", action="store_true",
                    help="H3: dedup on comment-normalized code (collapses blocks differing only in prose)")
    args = ap.parse_args()
    order = json.loads(Path(args.order).read_text())
    root = Path(args.aspire_root)
    per_ep, miss, lenmis, keymis = [], 0, 0, 0
    for ep in order:
        d = root / ep["subset"] / f"vlm_s2_subtask_fused_{args.tag}"
        prog, lines, trace = d / "program.py", d / "episode_000_lines.json", d / "trace.json"
        if args.require_skills_key and trace.exists():
            if json.loads(trace.read_text()).get("skills_key") != args.require_skills_key:
                keymis += 1; miss += 1; per_ep.append([""] * ep["length"]); continue
        if prog.exists() and lines.exists():
            fc = episode_frame_codes(prog, lines)
            if len(fc) != ep["length"]:
                lenmis += 1
                fc = (fc + [""] * ep["length"])[:ep["length"]]   # pad/trim to the dataset length (B3 rail)
        else:
            miss += 1; fc = [""] * ep["length"]
        per_ep.append(fc)
    codes, ids = assemble_flat_cache(per_ep, strip_comments=args.strip_comments)
    torch.save({"codes": codes, "code_id_by_idx": torch.from_numpy(ids)}, args.out)
    Path(args.out).with_name("episode_order.json").write_text(json.dumps(order))
    print(f"SAVED {args.out}: episodes={len(order)} frames={len(ids)} codes={len(codes)} "
          f"missing={miss} length_mismatch={lenmis} skills_key_mismatch={keymis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
