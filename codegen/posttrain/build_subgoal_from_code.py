"""Derive a LANGUAGE-ONLY subgoal cache from a code-GT cache (code-expert ablation).

The ablation asks whether the code expert needs executable code at all, or whether a short natural
the text differs -- otherwise the comparison confounds supervision content with slice boundaries.

Each code here is a commented program whose '# step k/n: <text>' line already states the subgoal in
words; that line becomes the whole slice text (prefix stripped). Codes without such a line fall back
to their first '# ' comment, and anything left is reported rather than guessed.
Output format is the standard flat cache: {"codes": [str], "code_id_by_idx": Long}.
"""
from __future__ import annotations
import argparse, json, re, collections
import torch

STEP = re.compile(r'^#\s*step\s+\d+\s*/\s*\d+\s*:\s*(.+)$', re.M)
FIRST_COMMENT = re.compile(r'^#\s*(?!goal:|bowl now:|plan:)(.+)$', re.M)


def subgoal_of(code: str):
    m = STEP.search(code)
    if m:
        return m.group(1).strip().rstrip('.')
    m = FIRST_COMMENT.search(code)
    return m.group(1).strip().rstrip('.') if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    c = torch.load(a.code_cache, map_location="cpu")
    codes, ids = c["codes"], c["code_id_by_idx"]

    texts, remap, miss = [""], {0: 0}, 0
    index = {}
    for i, code in enumerate(codes):
        if i == 0:
            continue
        s = subgoal_of(code)
        if s is None:
            miss += 1
            s = "continue the task"
        if s not in index:
            index[s] = len(texts)
            texts.append(s)
        remap[i] = index[s]
    new = torch.tensor([remap[int(v)] for v in ids], dtype=torch.long)
    cen = collections.Counter(texts[int(v)] for v in new)
    audit = {"code_codes": len(codes) - 1, "subgoal_codes": len(texts) - 1, "frames": int(new.numel()),
             "empty": int((new == 0).sum()), "no_step_line": miss,
             "top": [t for t, _ in cen.most_common(5)]}
    print(json.dumps(audit))
    if a.write:
        torch.save({"codes": texts, "code_id_by_idx": new}, a.out)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
