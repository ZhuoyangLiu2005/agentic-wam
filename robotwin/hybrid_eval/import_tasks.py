"""Import frozen programs into configs/robotwin/tasks/<task>/ (frozen_program.py + preamble.txt + task.yaml).

    python robotwin/hybrid_eval/import_tasks.py --frozen-root codegen/sim/frozen [--tasks a b ...]
        [--out configs/robotwin/tasks]

--frozen-root reads the codegen/sim program handoff (docs/ROBOTWIN.md):
frozen/<task>/{frozen_program.py, frozen_program.sha256, STAGE2_RESULT.md, task.yaml}. The program is the one
the frozen task.yaml names; it is refused if its sha256 differs from the recorded one, and the Stage-2 result
(clean/randomized/total, verdict, source) is taken from that task.yaml as recorded at freeze time.

--programs-root reads the older layout (`<task>/v<N>/frozen_program.py`, else a bare `frozen_program.py`,
with Stage-2 trial counts from --scores-root).

For every task the program is copied, its prompt preamble is written by the slicing engine (`slicing.py preamble`
form), and task.yaml records the sha256, the per-stage token budget (`gen_tokens`: the longest slice plus a
margin, in SDAR tokens) and the matching `nis`. `[stage each]` loops over a literal constant are expanded per
element for the token count, exactly as the engine expands them from the alignment records at build time.
Existing `short` names are kept from the previous tasks directory when one is given (--shorts-from).
"""
import argparse
import ast
import hashlib
import math
import re
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataprep.slicing import Program  # noqa: E402
from hybrid_eval import config  # noqa: E402

TOKEN_MARGIN = 8       # tokens beyond the longest slice (end-of-text + a little slack)
TOKEN_ROUND = 8        # canvas sizes are kept to multiples of this
# DEPRECATED (2026-09-16). `nis` reaches EVALUATION.num_inference_steps, which is the video/action
# denoiser's step count -- unrelated to the code canvas this scales it by. The ratio came from one
# configuration that happened to pair nis 40 with a 96-token canvas; that coincidence is not a relation.
# launch.py no longer reads the value; it is still written so the files stay diffable upstream.
NIS_PER_TOKEN = 40 / 96


def latest_program(task_dir):
    versions = sorted(task_dir.glob("v*/frozen_program.py"),
                      key=lambda p: int(re.fullmatch(r"v(\d+)", p.parent.name).group(1)))
    if versions:
        return versions[-1], versions[-1].parent.name
    bare = task_dir / "frozen_program.py"
    return (bare, "base") if bare.is_file() else (None, None)


def frozen_program(task_dir):
    """codegen/sim handoff: (program path, version label, frozen task.yaml) -- or (None, reason, None)."""
    meta_p = task_dir / "task.yaml"
    if not meta_p.is_file():
        return None, "no task.yaml", None
    meta = yaml.safe_load(meta_p.read_text()) or {}
    src = task_dir / meta.get("program", "frozen_program.py")
    if not src.is_file():
        return None, f"program {src.name} missing", None
    want = meta.get("sha256")
    got = hashlib.sha256(src.read_bytes()).hexdigest()
    if want and got != want:
        return None, f"sha256 mismatch (task.yaml {want[:12]} vs file {got[:12]})", None
    return src, "frozen", meta


def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(config.REPO / "checkpoints" / "SDAR-1.7B-Chat"))


def slice_texts(prog):
    """One slice per stage instance, `[stage each]` loops expanded over a literal iterable when they have one."""
    consts = prog._consts()
    plan = []
    for k in range(1, prog.n_stages + 1):
        if k not in prog.each:
            plan.append((prog.titles.get(k, ""), prog.stage_stmts(k)))
            continue
        loop = prog.stage_stmts(k)[0]
        targets = prog._for_targets(loop)
        it = loop.iter
        values = None
        if isinstance(it, ast.Name) and isinstance(consts.get(it.id), (list, tuple)):
            values = list(consts[it.id])
        elif isinstance(it, (ast.List, ast.Tuple)):
            try:
                values = list(ast.literal_eval(it))
            except (ValueError, SyntaxError):
                values = None
        if values is None or len(targets) != 1:
            plan.append((prog.titles.get(k, ""), (loop,)))   # unexpanded: counted as it stands
            continue
        for v in values:
            binding = {targets[0]: v}
            plan.append((prog._each_title(k, binding), prog._each_body(k, binding)))
    n = len(plan)
    out = []
    for i, (title, body) in enumerate(plan, 1):
        text = body if isinstance(body, str) else _unparse(prog, body)
        out.append(f"# [stage {i}/{n}] {title}\n{text}\n")
    return out


def _unparse(prog, stmts):
    from dataprep.slicing import unparse_stmts
    return unparse_stmts(list(stmts), prog.keep_calls)


def stage2_score(scores_root, task):
    if scores_root is None:
        return None
    parts = {}
    for split in ("clean", "randomized"):
        trials = list((scores_root / task / "oracle" / split).glob("trial_*"))
        if trials:
            parts[split] = f"{sum('reward_1.000' in t.name for t in trials)}/{len(trials)}"
    return parts or None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src_root = ap.add_mutually_exclusive_group(required=True)
    src_root.add_argument("--frozen-root", type=Path, help="codegen/sim program handoff: <root>/<task>/{frozen_program.py, task.yaml, ...}")
    src_root.add_argument("--programs-root", type=Path, help="older layout: <root>/<task>/v<N>/frozen_program.py")
    ap.add_argument("--tasks", nargs="*", help="task names (default: every directory with a frozen program)")
    ap.add_argument("--out", type=Path, default=None, help="tasks root (default: configs/robotwin/tasks)")
    ap.add_argument("--scores-root", type=Path, default=None, help="Stage-2 out_root with <task>/oracle/{clean,randomized}")
    ap.add_argument("--shorts-from", type=Path, default=None, help="previous tasks root whose task.yaml `short` names are kept")
    ap.add_argument("--source-prefix", default="codegen/sim/programs", help="how task.yaml names the program's path")
    a = ap.parse_args()
    out = a.out or config.tasks_root()
    tok = _tokenizer()
    root = a.frozen_root or a.programs_root
    pick = (lambda d: frozen_program(d)) if a.frozen_root else (lambda d: (*latest_program(d), None))
    names = a.tasks or sorted(d.name for d in root.iterdir() if d.is_dir() and pick(d)[0])
    rows = []
    for name in names:
        src, version, fmeta = pick(root / name)
        if src is None:
            print(f"{name}: {version or 'no frozen program'}, skipped"); continue
        prog = Program(src)
        errs = prog.check()
        if errs:
            print(f"{name}: contract errors, skipped:\n  " + "\n  ".join(errs)); continue
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, d / "frozen_program.py")
        pre = prog.prompt_preamble()
        (d / "preamble.txt").write_text(pre)
        slices = slice_texts(prog)
        n_tok = [len(tok.encode(s, add_special_tokens=False)) for s in slices]
        max_tok = max(n_tok)
        gen_tokens = int(math.ceil((max_tok + TOKEN_MARGIN) / TOKEN_ROUND) * TOKEN_ROUND)
        nis = int(round(gen_tokens * NIS_PER_TOKEN))
        short = name
        prev = (a.shorts_from / name / "task.yaml") if a.shorts_from else None
        if prev and prev.is_file():
            short = (yaml.safe_load(prev.read_text()) or {}).get("short", name)
        meta = {
            "short": short,
            "gen_tokens": gen_tokens,
            "nis": nis,
            "source": {
                "frozen_program": (f"frozen/{name}/{src.name}" if fmeta is not None
                                   else f"{a.source_prefix}/{name}/{version}/frozen_program.py" if version != "base"
                                   else f"{a.source_prefix}/{name}/frozen_program.py"),
                "version": version,
                **({"program_sha256": fmeta.get("sha256"), "embodiment": fmeta.get("embodiment"),
                    "record_instructions": fmeta.get("record_instructions")} if fmeta is not None else {}),
                "preamble": "preamble.txt (slicing.py preamble form: stage-called helper stubs + setup names)",
            },
            "stage_contract": {
                "annotated": True,
                "stages": prog.n_stages,
                "executed_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            },
            "tokens": {
                "preamble": len(tok.encode(pre, add_special_tokens=False)),
                "slices": n_tok,
                "max_slice": max_tok,
            },
        }
        score = (fmeta.get("stage2") if fmeta is not None else stage2_score(a.scores_root, name))
        if score:
            meta["stage2"] = score
        (d / "task.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, width=110))
        rows.append((name, version, prog.n_stages, len(slices), meta["tokens"]["preamble"], max_tok, gen_tokens, nis, score))
    print(f"{'task':28s} {'ver':5s} {'st':>2s} {'sl':>2s} {'pre':>4s} {'max':>4s} {'gen':>4s} {'nis':>3s}  stage2")
    for r in rows:
        print(f"{r[0]:28s} {r[1]:5s} {r[2]:2d} {r[3]:2d} {r[4]:4d} {r[5]:4d} {r[6]:4d} {r[7]:3d}  {r[8] or ''}")
    print(f"{len(rows)} tasks written under {out}")


if __name__ == "__main__":
    main()
