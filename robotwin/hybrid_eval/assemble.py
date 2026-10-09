#!/usr/bin/env python3
"""Code-slice toolbox shared by the online hybrid arbiter and the offline assembler.

Online (arbiter.py): parse_header, repair, whitelist_ok, allowed_names,
skeleton + setup_block (the prelude). Offline main: bucket a trace's slices by
header, majority-vote each step, repair, gate, and assemble a full program
(frozen-program skeleton + preamble setup + one slice per step).
Task-agnostic: the task enters through AGENTICWAM_FROZEN_PROGRAM / AGENTICWAM_CODE_PREAMBLE_FILE.

Usage: python robotwin/hybrid_eval/assemble.py --trace <trace-file...> --out <dir> --tag <name>
Emits <out>/<tag>.py + a JSON report line on stdout.
"""
import argparse
import ast
import builtins
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# Task-parametrised: the assembler is the single source of the prelude, the
# whitelist and the repair, so pointing these two at a task's assets is all
# that is needed to assemble/repair that task's slices. There is deliberately
# NO default: a missing variable must fail loudly, never silently fall back to
# another task's whitelist/prelude.
#   AGENTICWAM_FROZEN_PROGRAM  - frozen program the skeleton + whitelist come from
#   AGENTICWAM_CODE_PREAMBLE_FILE - the preamble the eval already exports


def _asset(var):
    p = os.environ.get(var, "")
    if not p or not os.path.isfile(p):
        raise RuntimeError(f"{var} must point at the task's asset file (got {p!r})")
    return Path(p)


def frozen_program():
    return _asset("AGENTICWAM_FROZEN_PROGRAM")


def preamble():
    return _asset("AGENTICWAM_CODE_PREAMBLE_FILE")
# Stage header, the first line of every slice the model emits: `# [stage k/N] ...` (`step` before 2026-09-06).
# The prompt may end inside the header, in which case the completion opens with
# the bare `k/N] ...`. N comes from the header itself, so nothing here is per task.
_HDR = re.compile(r"(?:stage|step)\s*(\d+)\s*/\s*(\d+)", re.I)   # `stage` = 2026-09-06 contract; `step` = older ckpts
_BARE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*\]")


def parse_header(code):
    """First line of a slice -> (k, N, how) with 1 <= k <= N, else (None, None, 'none')."""
    lines = code.splitlines()
    first = lines[0] if lines else ""
    for rx, how in ((_HDR, "header"), (_BARE, "bare_header")):
        m = rx.search(first)
        if m:
            k, n = int(m.group(1)), int(m.group(2))
            # Trust k only: the joint multi-task model sometimes writes the wrong denominator
            # ("# [step 3/2]" on a perfect step-3 slice), so N is informational and must not gate the step.
            if k >= 1:
                return k, n, how
    return None, None, "none"


def skeleton():
    """Imports + UPPERCASE constants + helper defs from the frozen program.

    Which constants survive is decided by dataprep.slicing.hoisted_consts, the SAME call
    _main_stmts uses to decide what to leave in the main body -- so a constant this drops is
    emitted by the setup block instead of vanishing from the prelude entirely.
    """
    from dataprep.slicing import hoisted_consts, _module_aliases, is_const   # one rule for slicing and deploy
    tree = ast.parse(frozen_program().read_text(encoding="utf-8"))
    kept = hoisted_consts(tree)
    keep = [st for st in tree.body
            if isinstance(st, (ast.Import, ast.ImportFrom, ast.FunctionDef)) or id(st) in kept]
    return "\n".join(ast.unparse(st) for st in keep)


def setup_block():
    """The frozen program's `[setup]` statements as executable code (print/guard-stripped, same rendering as the
    slices); run once after the skeleton, before the first generated slice."""
    from dataprep.slicing import Program
    # keep_guards=True: the frozen programs defend their own perception reads with
    # `if pose is None: raise RuntimeError("could not ground ...")`. Offline slicing drops those
    # (they never fire on successful demos); the LIVE prelude needs them, because a stripped guard
    # let the next statement index None and report a bare "TypeError: 'NoneType' object is not
    # subscriptable" instead of naming the object that failed to ground. This is a DIAGNOSTIC
    # change: it does not change how often the prelude fails, only how legibly it fails.
    return Program(frozen_program()).exec_setup(keep_guards=True).strip()


def compiles(code):
    try:
        compile(code, "<gen>", "exec")
        return True
    except SyntaxError:
        return False


_KEYWORD_LINES = {"else:", "try:", "finally:", "continue", "break", "pass"}


def repair(code, allow_r3=False):
    """Comment out (never delete) every line that blocks compilation. Each
    neutralized line is classified 'debris' (a bare non-keyword identifier or
    a header fragment with no '='/'(' — clearly not a statement) or 'code'
    (looked like a real statement). Tiers: R0 untouched; R1 only debris
    touched; R2 at least one code-like line touched (the online gate rejects
    R2). Returns (code, tier) or (None, 'unrepairable'). A slice that already
    compiles is returned verbatim."""
    lines = code.splitlines()
    touched = []

    def classify(ln):
        cp = ln.split("#", 1)[0].strip()
        if cp in _KEYWORD_LINES or cp.startswith(("elif ", "except")):
            return "code"
        if not cp or re.fullmatch(r"[A-Za-z_]\w*", cp) or ("=" not in cp and "(" not in cp):
            return "debris"
        return "code"

    for _ in range(len(lines) + 1):
        try:
            compile("\n".join(lines), "<gen>", "exec")
            break
        except SyntaxError as e:
            k = (e.lineno or 1) - 1
            if not (0 <= k < len(lines)):
                return None, "unrepairable"
            if lines[k].lstrip().startswith("#"):
                # Python is blaming a line this loop already commented. That happens exactly
                # when a compound header outlives its whole body ("expected an indented block
                # after 'if' statement on line N"): the header is valid on its own so it is
                # never blamed, and re-commenting the body is a no-op, so the loop span until
                # it ran out of iterations and returned unrepairable. Measured: 218 of 218
                # unrepairable generations exited this way. Comment the HEADER instead.
                j = k - 1
                while j >= 0 and not (lines[j].rstrip().endswith(":")
                                      and not lines[j].lstrip().startswith("#")):
                    j -= 1
                if j < 0:
                    return None, "unrepairable"
                k = j
                touched.append("code")      # dropping a block header changes control flow
                lines[k] = "# " + lines[k]
                continue
            touched.append(classify(lines[k]))
            lines[k] = "# " + lines[k]
    out = "\n".join(lines)
    if not compiles(out):
        return None, "unrepairable"
    if not touched:
        return out, "R0"
    tier = "R2" if "code" in touched else "R1"
    return out, f"{tier}:d{touched.count('debris')}c{touched.count('code')}"


# Pure, read-only names a slice may call besides the task's own helpers: numpy /
# builtin computation and three exception types. None of these can move the robot,
# so they do not weaken the gate. Single definition for the online arbiter and
# every offline tool (assembler main, codegen_decode_sweep).
PURE_ALLOWED = frozenset({
    "np", "array", "asarray", "clip", "float", "int", "abs", "len", "round", "min", "max",
    "print", "range", "enumerate", "zip", "list", "dict", "tuple", "str", "bool", "sum",
    "any", "all", "sorted", "set", "isinstance", "linalg", "norm", "dot", "cross", "zeros", "ones",
    "sqrt", "deg2rad", "rad2deg", "RuntimeError", "ValueError", "Exception"})


def allowed_names(extra=()):
    """Callable names a slice may use: every function the frozen program defines,
    every helper stub the preamble declares (`def name(...)`), PURE_ALLOWED, plus
    `extra` (the online arbiter adds the task API's callables at bind time; an
    offline tool that has no live API passes them here explicitly)."""
    names = {n.name for n in ast.walk(ast.parse(frozen_program().read_text(encoding="utf-8"))) if isinstance(n, ast.FunctionDef)}
    names |= set(re.findall(r"^def\s+([A-Za-z_]\w*)\s*\(", preamble().read_text(encoding="utf-8"), re.M))
    return names | set(PURE_ALLOWED) | set(extra)


_DENY_ROOTS = {"os", "sys", "subprocess", "importlib", "builtins", "shutil", "socket",
               "pathlib", "task_env", "env", "self", "torch", "__builtins__"}


def whitelist_ok(code, allowed):
    """Safety gate for a generated slice. Plain calls `f(...)` must name an
    allowed function (task API, frozen-program helpers, preamble stubs, pure
    builtins). Attribute calls `obj.m(...)` are allowed unless the root object
    is a system/env handle or the attribute is a dunder (`np.linalg.norm`,
    `lst.append` pass; `os.system`, `task_env.x` fail). Imports never pass."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name):
            if f.id not in allowed:
                return False
        elif isinstance(f, ast.Attribute):
            if f.attr.startswith("__"):
                return False
            root = f
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in _DENY_ROOTS:
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--r3", action="store_true")
    a = ap.parse_args()

    codes = []
    for f in a.trace:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            try:
                codes.append(json.loads(line).get("code", ""))
            except json.JSONDecodeError:
                pass

    # --- selection: header bucket + exact-string majority vote (header only)
    buckets, n_steps, unheadered = {}, 0, 0
    for c in codes:
        k, n, _ = parse_header(c)
        if k is None:
            unheadered += 1
            continue
        n_steps = max(n_steps, n)
        buckets.setdefault(k, []).append(c)

    sk = skeleton()
    allowed = allowed_names()

    report = {"tag": a.tag, "n_replans": len(codes), "n_steps": n_steps,
              "unheadered": unheadered, "steps": {}}
    parts = [sk, "", setup_block(), ""]
    ok = n_steps > 0
    for k in range(1, n_steps + 1):
        if not buckets.get(k):
            report["steps"][k] = {"status": "missing"}
            ok = False
            continue
        vote, cnt = Counter(buckets[k]).most_common(1)[0]
        fixed, tier = repair(vote, allow_r3=a.r3)
        if fixed is None:
            report["steps"][k] = {"status": "unrepairable", "votes": cnt}
            ok = False
            continue
        if not whitelist_ok(fixed, allowed):
            report["steps"][k] = {"status": "rejected", "tier": tier, "votes": cnt}
            ok = False
            continue
        report["steps"][k] = {"status": "ok", "tier": tier.split(":")[0],
                              "lines_commented": tier.split(":")[1] if ":" in tier else "",
                              "votes": cnt, "bucket": len(buckets[k])}
        parts.append(f"# --- generated step {k} (tier {tier}, votes {cnt}/{len(buckets[k])}) ---")
        parts.append(fixed)
        parts.append("")
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    prog = "\n".join(parts)
    (outdir / f"{a.tag}.py").write_text(prog)
    report["assembled"] = str(outdir / f"{a.tag}.py")
    report["program_compiles"] = compiles(prog)
    report["complete"] = ok
    print(json.dumps(report))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
