"""Build the code expert's skill-conditioning block: {instruction: skill_block}.

Mirrors the RoboTwin EXP-1 artifact (`task_skill_prompt_by_instruction.json`), which the dataset
consumes as `skill_prompt_map` and emits as `prompt_code = skill_block + instruction`. `agenticwam_model`
feeds `prompt_code` to the CODE expert only — video and action keep the vanilla prompt — so this
conditions code generation without touching the other two modalities.

WHY IT IS CHEAP: `prompt_code` goes to `code_expert.build_code_batch()`, i.e. it is tokenized by the
SDAR tokenizer. It does NOT go through the Cosmos text encoder, so no new prompt-embeds cache is
needed and the 128-token Cosmos canvas does not apply. The only budget is `max_code_tokens`, shared
with the code itself.

WHY IT IS RETRIEVED, NOT THE WHOLE LIBRARY: all 27 frozen skills would be ~6-7k chars (~2k tokens)
and would crowd out the code the expert must actually produce. RoboTwin includes only the skills
relevant to the task (median 733 chars). We select by evidence: the skills the r2 actor actually
called on this dataset, plus a small hand-picked set that the task description implies.

COMPRESSION: signature + first docstring line + body, comments and blank lines dropped — the same
shape RoboTwin uses. The body is kept because it is what tells the expert which core verbs a skill
expands into, which is exactly the knowledge the event-contract grounding depends on.
"""
from __future__ import annotations

import argparse
import ast
import json
import os

HEADER = "# Skills available for this task (call as plain functions, robot first):"


def compress(skill_py: str, max_doc: int = 90) -> str | None:
    """One skill -> `def sig:` + one-line docstring + body, minus comments/blank lines."""
    tree = ast.parse(skill_py)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None:
        return None
    args = ", ".join(a.arg for a in fn.args.args)
    out = [f"def {fn.name}({args}):"]
    doc = ast.get_docstring(fn)
    if doc:
        first = " ".join(doc.strip().splitlines()[0].split())[:max_doc]
        out.append(f'    """{first}"""')
    src = skill_py.splitlines()
    for stmt in fn.body:
        if isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, "value", None), ast.Constant) \
                and isinstance(stmt.value.value, str):
            continue                                   # the docstring, already emitted
        for ln in src[stmt.lineno - 1:stmt.end_lineno]:
            s = ln.rstrip()
            if s.strip() and not s.strip().startswith("#"):
                out.append(s if s.startswith("    ") else "    " + s.strip())
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--skills", required=True, help="comma-separated skill dir names (kebab-case)")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="", help="optional HF id/path to count exact tokens")
    a = ap.parse_args()

    blocks = [HEADER]
    for name in a.skills.split(","):
        name = name.strip()
        p = os.path.join(a.lib, name, "skill.py")
        if not os.path.exists(p):
            raise SystemExit(f"no such skill: {p}")
        c = compress(open(p).read())
        if c:
            blocks.append(c)
    block = "\n".join(blocks) + "\n"

    json.dump({a.instruction: block}, open(a.out, "w"), indent=1)
    print(f"[skill-prompt] {len(a.skills.split(','))} skills, {len(block)} chars -> {a.out}")

    if a.tokenizer:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
        n_block = len(tk(block)["input_ids"])
        n_full = len(tk(block + a.instruction)["input_ids"])
        print(f"[tokens] skill block {n_block} | block+instruction {n_full}")
    print(block[:400])
    print("SKILL_PROMPT_DONE")


if __name__ == "__main__":
    main()
