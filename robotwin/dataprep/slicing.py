#!/usr/bin/env python3
"""Stage contract + slicing engine: turns an ANNOTATED frozen program (plus the collection's alignment
records) into the code-stream training assets — preamble, per-stage slices, per-episode stage frame ranges.

The contract (comment lines in the program's main body; nothing else is task-specific):

    # [setup]                     following top-level statements form the prelude's `# Setup:` block
    # [stage k/N] <title>         following top-level statements form stage k's slice (k = 1..N, contiguous)
    # [stage each] <title>        the next statement is a top-level `for`; every EXECUTED iteration becomes one
                                  stage of its own (one grasp-release cycle per object). The loop targets are
                                  bound to that iteration's values in the slice text; the title is a template
                                  evaluated over the program's constants and the loop variables, e.g.
                                  `# [stage each] pick and place the {NAMES[idx]} block`. Counts as one `k`
                                  in the fixed numbering; the per-episode N includes the executed iterations.
    # [omit]                      following top-level statements are left out (dead fallbacks, audits)

Sections may be reopened (e.g. `# [setup]` again after stage 1). Imports, UPPERCASE constants and `def`s
are always prelude material wherever they sit. A helper's return description is the first line of its
docstring, or an explicit string return annotation `def f(...) -> "desc":` which takes precedence
(emitted as `def f(args) -> "desc": ...` in the preamble); helpers without either get a bare stub.

Frames: a stage's frame range starts at the first motion record that belongs to it and ends where the next
stage's first record starts (stage 1 starts at frame 0; the last stage ends at the episode's last frame).
A record belongs to the stage whose top-level statement issued it: `top_line` (the outermost program
frame, recorded by the collector) when present; for `[stage each]` loops the iteration is identified by the
loop targets' values in `top_locals` (the outermost frame's simple-typed locals, also recorded by the
collector). Older alignment files only carry `line` (the innermost
frame); a record raised inside a helper is then attributed to the earliest stage (not before the current
one) whose code calls that helper, transitively — ambiguous when consecutive stages share a helper, so
such files get a warning and the frame ranges should not be trusted for training.

CLI:
    slicing.py check    <task | program.py> [--executed frozen.py]   contract + compile check; for a task also
                                                                    verifies tasks/<task>/preamble.txt is current
    slicing.py preamble <task | program.py> [--write]               print (or write tasks/<task>/preamble.txt)
    slicing.py build    --task T --align-dir D --out O [--n-train 50] [--header-word stage|step]
    slicing.py regress  --task T --align-dir D --legacy <old cache>  byte-compare with a legacy v7s cache

Cache layout written by `build` (what a data config points at):
    <out>/preamble.txt                 prompt text: stage-called helper signatures + `# Setup defines:` (also the
                                       deploy-time AGENTICWAM_CODE_PREAMBLE_FILE; the eval derives the executable setup
                                       from the frozen program, see exec_setup)
    <out>/slices_ep/ep_%06d.json       {"preamble": str, "slices": [str, ...]} per kept episode
    <out>/cycle_by_idx.pt              {"cycle_by_idx": LongTensor[total frames] (stage index), "action_size": 14}
    <out>/episodes_{train,val,all}.json  episode-index lists (kept episodes sorted by seed; train = first n)
    <out>/stage_frames.json            per-episode stage frame ranges (the slicing catalog)
"""
import argparse
import ast
import builtins
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid_eval import config  # noqa: E402

_MARK = re.compile(r"^\s*#\s*\[(setup|omit|stage)\s*(?:(\d+)\s*/\s*(\d+)|(each))?\]\s*(.*?)\s*$")
_HEADER_RX = re.compile(r"(stage|step)\s*(\d+)\s*/\s*(\d+)", re.I)


# ----------------------------------------------------------------------------- AST helpers (v7s contract)
class _Unparser(getattr(ast, "_Unparser", object)):
    """ast.unparse with tuples always parenthesised (`(a, b) = f()`), the Python 3.10 rendering the
    existing caches and checkpoints were produced with. Python 3.11+ drops those parentheses, so the
    stock unparser would make preambles/slices differ by interpreter version."""
    def visit_Tuple(self, node):
        with self.delimit("(", ")"):
            self.items_view(self.traverse, node.elts)


def unparse(node):
    if hasattr(ast, "_Unparser"):
        return _Unparser().visit(node)
    return ast.unparse(node)


PRINT_LIKE = {"print"}   # extended per program with helpers whose body only prints (`def P(*a): print(*a)`)


def is_print(st):
    return (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)
            and isinstance(st.value.func, ast.Name) and st.value.func.id in PRINT_LIKE)


def print_only_helpers(tree):
    """Names of module-level defs whose body is nothing but print statements (docstring / pass / bare
    `return` allowed): calls to them are logging, stripped like `print()` itself."""
    def trivial(st):
        return (is_print(st) or isinstance(st, ast.Pass) or _is_docstring(st)
                or (isinstance(st, ast.Return) and st.value is None))
    return {fn.name for fn in tree.body if isinstance(fn, ast.FunctionDef)
            and fn.body and all(trivial(st) for st in fn.body) and any(is_print(st) for st in fn.body)}


class _Strip(ast.NodeTransformer):
    """Drop print() statements everywhere; drop `if ...: raise` guards (they never fire in kept episodes) —
    unless the guard's test itself calls a helper or a motion API (`if not reach(...): raise`): that call is
    part of the executed behaviour and must stay in the slice text."""
    def __init__(self, keep_calls=(), keep_guards=False):
        super().__init__()
        self.keep_calls = set(keep_calls)
        # keep_guards=True disables the `if ...: raise` stripping below. The "they never fire"
        # premise holds for OFFLINE slicing of successful demo episodes (what this transformer
        # was written for) but NOT for the hybrid eval's LIVE prelude, where the task's own
        # `if pose is None: raise RuntimeError(...)` is exactly the check that should fire when
        # perception misses. Stripping it turned an actionable RuntimeError naming the
        # ungrounded object into a TypeError two statements later. Only
        # hybrid_eval.assemble.setup_block() sets this; every other caller keeps the historical
        # behaviour, so the training-data slicing path is byte-identical.
        self.keep_guards = keep_guards

    def _test_calls_kept(self, test):
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in self.keep_calls
                   for n in ast.walk(test))

    def _clean(self, body):
        out = []
        for s in body:
            if is_print(s):
                continue
            r = self.visit(s)
            if r is None:
                continue
            out.extend(r if isinstance(r, list) else [r])
        return out or [ast.Pass()]

    def visit_If(self, n):
        if (not self.keep_guards
                and all(isinstance(s, ast.Raise) or is_print(s) for s in n.body) and not n.orelse
                and not self._test_calls_kept(n.test)):
            return None
        n.body = self._clean(n.body)
        if n.orelse:   # an else made only of prints disappears, it does not become `else: pass`
            el = self._clean(n.orelse)
            n.orelse = [] if all(isinstance(x, ast.Pass) for x in el) else el
        return n

    def visit_For(self, n):
        n.body = self._clean(n.body); n.orelse = []; return n

    def visit_While(self, n):
        n.body = self._clean(n.body); return n

    def visit_Try(self, n):
        if all(all(is_print(s) or isinstance(s, ast.Continue) for s in h.body) for h in n.handlers):
            return self._clean(n.body)
        n.body = self._clean(n.body)
        for h in n.handlers:
            h.body = self._clean(h.body)
        return n


MOTION_CALLS = {"goto_pose", "close_gripper", "open_gripper", "goto_home_joint_position"}


def unparse_stmts(stmts, keep_calls=(), keep_guards=False):
    out = []
    for st in stmts:
        if is_print(st):
            continue
        r = _Strip(keep_calls, keep_guards).visit(st)
        if r is None:
            continue
        for x in (r if isinstance(r, list) else [r]):
            ast.fix_missing_locations(x)
            out.append(unparse(x))
    return "\n".join(out)


def _module_aliases(tree):
    """Names bound by the program's imports (`np`, `math`, ...)."""
    out = set()
    for s in tree.body:
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in s.names}
    return out


def is_const(s, aliases=frozenset()):
    """UPPERCASE assignment whose value is built only from literals, other constants, imports and builtins
    (`Q_DOWN = np.array(...)`, `HOVER = LIFT + 0.02`). `R0 = quat_matrix(mic_quat)` is main-body code:
    hoisting it would run it before the setup that defines its inputs."""
    if not isinstance(s, ast.Assign):
        return False
    names = []
    for t in s.targets:
        names += list(t.elts) if isinstance(t, ast.Tuple) else [t]
    if not all(isinstance(n, ast.Name) and n.id.isupper() for n in names):
        return False
    return all(n.id.isupper() or n.id in aliases or n.id in _BUILTINS
               for n in ast.walk(s.value) if isinstance(n, ast.Name))


_BUILTINS = set(dir(__builtins__)) if isinstance(__builtins__, dict) is False else set(__builtins__)


def _return_desc(fn):
    """Text after `->` in the stub: a string return annotation (`-> "desc"`) wins; a real type annotation
    (`-> bool`) is kept verbatim; else the docstring's first line; else nothing."""
    if isinstance(fn.returns, ast.Constant) and isinstance(fn.returns.value, str):
        return json.dumps(fn.returns.value)
    if fn.returns is not None:
        return unparse(fn.returns)
    doc = ast.get_docstring(fn)
    if doc:
        return json.dumps(doc.strip().splitlines()[0].strip())
    return None


# ----------------------------------------------------------------------------- contract
def _is_docstring(s):
    return isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)


def hoisted_consts(tree):
    """The is_const statements the skeleton actually KEEPS, as a set of statement ids.

    is_const judges each statement alone, so it can accept one whose free names it rejected
    earlier: in hanging_mug `_Z2T = q2R([...])` is rejected (a call to a lowercase local helper)
    while `_FLIP = _Z2T.T @ np.diag(...) @ _Z2T` is accepted. The skeleton then prunes `_FLIP` to a
    fixpoint because `_Z2T` is unbound -- but _main_stmts had already dropped `_FLIP` from the main
    body on the strength of is_const alone, so it landed in NEITHER half of the prelude and
    delta_R raised `NameError: _FLIP` the moment a stage called it. Both sides must ask the same
    question, so they both ask it here.
    """
    al = _module_aliases(tree)
    keep = [s for s in tree.body
            if isinstance(s, (ast.Import, ast.ImportFrom, ast.FunctionDef)) or is_const(s, al)]

    def _binds(st):
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            return {(a.asname or a.name).split(".")[0] for a in st.names}
        if isinstance(st, ast.FunctionDef):
            return {st.name}
        return {n.id for t in getattr(st, "targets", []) for n in ast.walk(t)
                if isinstance(n, ast.Name)}

    def _reads(st):
        if isinstance(st, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            return set()        # a function body resolves at call time, not here
        return {n.id for n in ast.walk(st)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

    bi = set(dir(builtins))
    while True:
        bound = bi | {n for st in keep for n in _binds(st)}
        pruned = [st for st in keep if not (_reads(st) - bound)]
        if len(pruned) == len(keep):
            break
        keep = pruned
    return {id(s) for s in keep if not isinstance(s, (ast.Import, ast.ImportFrom, ast.FunctionDef))}


def _main_stmts(tree):
    # A constant the skeleton drops stays in the main body, where it keeps its source position and
    # is emitted by the setup block -- see hoisted_consts.
    kept = hoisted_consts(tree)
    return [s for s in tree.body
            if not isinstance(s, (ast.FunctionDef, ast.Import, ast.ImportFrom)) and id(s) not in kept
            and not _is_docstring(s)]


def _strip_returns(tree):
    t = ast.parse(unparse(tree))
    for n in ast.walk(t):
        if isinstance(n, ast.FunctionDef):
            n.returns = None
    return ast.dump(t)


def equivalent(annotated_src, executed_src):
    """True iff the two programs differ only in comments and helper return annotations."""
    return _strip_returns(ast.parse(annotated_src)) == _strip_returns(ast.parse(executed_src))


def _freeze(v):
    return tuple(_freeze(x) for x in v) if isinstance(v, (list, tuple)) else v


class Program:
    def __init__(self, path, executed_src=None):
        """`executed_src`: the program text the alignment records refer to (embedded in the alignment
        JSON). When it is not the annotated text itself, record lines are mapped statement-by-statement
        (same statement sequence, comments/annotations aside) — the check that annotating changed nothing."""
        self.path = Path(path)
        self.src = self.path.read_text()
        self.tree = ast.parse(self.src)
        self.exec_main, self.exec_helpers = None, None
        if executed_src is not None and executed_src != self.src:
            if not equivalent(self.src, executed_src):
                raise ValueError(f"{self.path} is not a comment/annotation-only variant of the executed program")
            et = ast.parse(executed_src)
            self.exec_main = _main_stmts(et)
            self.exec_helpers = [s for s in ast.walk(et) if isinstance(s, ast.FunctionDef)]
        self.lines = self.src.splitlines()
        self.helpers = [s for s in ast.walk(self.tree) if isinstance(s, ast.FunctionDef)]
        self.markers = []                       # (line, kind, k, n, title); kind "each" = repeated stage
        for i, l in enumerate(self.lines, 1):
            m = _MARK.match(l)
            if m:
                kind = "each" if m.group(4) else m.group(1)
                self.markers.append((i, kind, int(m.group(2)) if m.group(2) else None,
                                     int(m.group(3)) if m.group(3) else None, m.group(5)))
        self.main = _main_stmts(self.tree)
        if self.exec_main is not None:
            assert len(self.exec_main) == len(self.main) and len(self.exec_helpers) == len(self.helpers)
        self.sections = {}                      # statement -> ("setup" | "omit" | k)
        self.titles = {}
        self.each = {}                          # k -> title template (repeated stage: one per loop iteration)
        # `[stage each]` markers take the next free k in program order, so the fixed numbering stays 1..N.
        numbered, next_k = [], 1
        for (ln, kind, k, n, title) in self.markers:
            if kind == "each":
                k = next_k
            if kind in ("stage", "each"):
                next_k = max(next_k, (k or 0) + 1)
            numbered.append((ln, kind, k, n, title))
        self.markers = numbered
        for s in self.main:
            sec = "setup"
            for (ln, kind, k, n, title) in self.markers:
                if ln < s.lineno:
                    sec = kind if kind in ("setup", "omit") else k
                    if kind == "stage":
                        self.titles.setdefault(k, title)
                    elif kind == "each":
                        self.titles.setdefault(k, title); self.each[k] = title
            self.sections[s] = sec
        self.n_stages = max((k for (_, kind, k, _, _) in self.markers if kind in ("stage", "each")), default=0)
        self.print_only = print_only_helpers(self.tree)
        self._arm()

    def _arm(self):
        """Point the module-level stripping sets at this program (the last-constructed program wins;
        every entry point works on one program at a time)."""
        self.keep_calls = MOTION_CALLS | {fn.name for fn in self.helpers}
        PRINT_LIKE.clear(); PRINT_LIKE.add("print"); PRINT_LIKE.update(self.print_only)

    # --- validation ---------------------------------------------------------------------------------
    def check(self):
        errs = []
        stage_ks = [k for (_, kind, k, _, _) in self.markers if kind in ("stage", "each")]
        for k in self.each:
            stmts = self.stage_stmts(k)
            if len(stmts) != 1 or not isinstance(stmts[0], ast.For):
                errs.append(f"[stage each] {k} must be followed by exactly one top-level `for` statement "
                            f"(got {len(stmts)}: {[type(x).__name__ for x in stmts]})")
            elif not self._for_targets(stmts[0]):
                errs.append(f"[stage each] {k}: the loop target must be a name or a tuple of names")
        if not stage_ks:
            errs.append("no `# [stage k/N]` marker")
        if sorted(set(stage_ks)) != list(range(1, self.n_stages + 1)):
            errs.append(f"stage numbers must be contiguous 1..N, got {sorted(set(stage_ks))}")
        for (ln, kind, k, n, title) in self.markers:
            if kind == "stage" and n is not None and n != self.n_stages and not self.each:
                errs.append(f"line {ln}: N={n} but the program has {self.n_stages} stages")
            if kind in ("stage", "each") and not title:
                errs.append(f"line {ln}: stage {k} has no title")
        for k in range(1, self.n_stages + 1):
            if not any(v == k for v in self.sections.values()):
                errs.append(f"stage {k} has no statements")
        omitted_motion = [s.lineno for s in self.main if self.sections[s] == "omit"
                          and self._calls_in(s) & MOTION_CALLS]
        if omitted_motion:
            errs.append(f"omitted statements issue motions (lines {omitted_motion}): motions must belong to a stage")
        # every slice must compile on its own and after the prelude
        try:
            compile(self.preamble(), "<preamble>", "exec")
        except SyntaxError as e:
            errs.append(f"preamble does not compile: {e}")
        for k, text in enumerate(self.slices(), 1):
            try:
                compile(text, f"<stage{k}>", "exec")
            except SyntaxError as e:
                errs.append(f"stage {k} does not compile: {e}")
        for k in self.each:                     # loop bodies are only instantiated per episode: compile them once here
            stmts = self.stage_stmts(k)
            if stmts and isinstance(stmts[0], ast.For):
                try:
                    compile(unparse_stmts(stmts[0].body, self.keep_calls), f"<stage{k} body>", "exec")
                except SyntaxError as e:
                    errs.append(f"[stage each] {k} body does not compile: {e}")
        return errs

    # --- texts ---------------------------------------------------------------------------------------
    def setup_stmts(self):
        return [s for s in self.main if self.sections[s] == "setup"]

    def stage_stmts(self, k):
        return [s for s in self.main if self.sections[s] == k]

    def preamble(self):
        lines = [unparse(s) for s in self.tree.body if isinstance(s, (ast.Import, ast.ImportFrom))]
        lines.append("")
        # hoisted_consts, not is_const: a constant the skeleton drops now stays in the main body
        # and is printed under '# Setup:' below -- listing it here too would show it twice.
        _kept = hoisted_consts(self.tree)
        lines += [unparse(s) for s in self.tree.body if id(s) in _kept]
        lines.append("")
        for fn in self.helpers:
            ret = _return_desc(fn)
            lines.append(f"def {fn.name}({unparse(fn.args)})" + (f" -> {ret}" if ret else "") + ": ...")
        return "\n".join(lines) + "\n" + "\n# Setup:\n" + unparse_stmts(self.setup_stmts(), self.keep_calls) + "\n"

    def called_helpers(self):
        """Names of the helpers a stage body calls directly, in definition order (what a slice may reference);
        print-only helpers are left out, their calls are stripped from the slices like `print()`."""
        names = set()
        for k in range(1, self.n_stages + 1):
            for s in self.stage_stmts(k):
                names |= self._calls_in(s)
        return [fn.name for fn in self.helpers if fn.name in names and fn.name not in self.print_only]

    def setup_names(self):
        """Names the setup block binds that some stage body reads, in definition order."""
        used = set()
        for k in range(1, self.n_stages + 1):
            for s in self.stage_stmts(k):
                used |= {n.id for n in ast.walk(s) if isinstance(n, ast.Name)}
        out = []
        for s in self.setup_stmts():
            for node in ast.walk(s):
                if not isinstance(node, ast.Assign):
                    continue
                for t in node.targets:
                    elts = [t] if isinstance(t, ast.Name) else [e for e in getattr(t, "elts", []) if isinstance(e, ast.Name)]
                    for nm in elts:
                        if nm.id in used and nm.id not in out and not nm.id.startswith("_"):
                            out.append(nm.id)
        return out

    def prompt_preamble(self, style="sig", only_called=True, setup_names=True):
        """The `# Helper library:` text the code expert is prompted with (2026-09-10 contract): one stub per
        helper a stage calls -- signature only (`sig`), or with the return description (`desc`) -- and one
        line naming the setup variables the stages read. `full` is the legacy preamble (imports, constants,
        every stub with its description, the executable `# Setup:` block); `only_called=False` lists every
        helper. Nothing here is executed: the eval runs `exec_setup()` of the frozen program instead."""
        if style == "full":
            return self.preamble()
        keep = set(self.called_helpers()) if only_called else {fn.name for fn in self.helpers}
        lines = []
        for fn in self.helpers:
            if fn.name not in keep:
                continue
            ret = _return_desc(fn) if style == "desc" else None
            lines.append(f"def {fn.name}({unparse(fn.args)})" + (f" -> {ret}" if ret else "") + ": ...")
        if setup_names and self.setup_names():
            lines.append("# Setup defines: " + ", ".join(self.setup_names()))
        return "\n".join(lines) + "\n"

    def exec_setup(self, keep_guards=False):
        """The `[setup]` statements as executable code: what the hybrid eval runs (after the frozen program's
        skeleton) before the first generated slice."""
        return unparse_stmts(self.setup_stmts(), self.keep_calls, keep_guards) + "\n"

    @staticmethod
    def _for_targets(loop):
        if not isinstance(loop, ast.For):
            return []
        t = loop.target
        names = [t] if isinstance(t, ast.Name) else (list(t.elts) if isinstance(t, ast.Tuple) else [])
        return [n.id for n in names] if names and all(isinstance(n, ast.Name) for n in names) else []

    def _consts(self):
        """UPPERCASE constants with literal values: the namespace `[stage each]` titles may reference."""
        out = {}
        for s in self.tree.body:
            if is_const(s):
                try:
                    val = ast.literal_eval(s.value)
                except (ValueError, SyntaxError):
                    continue
                for t in s.targets:
                    names = list(t.elts) if isinstance(t, ast.Tuple) else [t]
                    vals = list(val) if isinstance(t, ast.Tuple) else [val]
                    out.update({n.id: v for n, v in zip(names, vals)})
        return out

    def _each_body(self, k, binding):
        """Loop body of `[stage each]` k with the loop targets replaced by this iteration's literals."""
        loop = self.stage_stmts(k)[0]
        text = unparse_stmts(loop.body, self.keep_calls)
        for name, val in binding.items():
            text = re.sub(rf"\b{re.escape(name)}\b", repr(val), text)
        return text

    def _each_title(self, k, binding):
        ns = dict(self._consts()); ns.update(binding)

        def ev(m):
            try:
                return str(eval(m.group(1), {"__builtins__": {}}, ns))   # noqa: S307 (title template, no builtins)
            except Exception:
                return m.group(0)
        return re.sub(r"\{([^{}]+)\}", ev, self.titles.get(k, ""))

    def _iterations(self, k, records):
        """Executed iterations of `[stage each]` k, in order of first appearance: [{target: value}]."""
        loop = self.stage_stmts(k)[0]; targets = self._for_targets(loop)
        if not targets:
            return []
        src_loop = (self.exec_main if self.exec_main is not None else self.main)[self.main.index(loop)]
        seen, keys = [], []
        for r in records:
            ln = r.get("top_line")
            if ln is None or not (src_loop.lineno <= int(ln) <= src_loop.end_lineno):
                continue
            loc = r.get("top_locals") or {}
            if not all(t in loc for t in targets):
                continue
            key = tuple(_freeze(loc[t]) for t in targets)
            if key not in keys:
                keys.append(key); seen.append({t: loc[t] for t in targets})
        return seen

    def stage_plan(self, records=None):
        """Ordered stage instances for one episode: [(k, title, body_text, binding | None)].
        Fixed stages appear once; a `[stage each]` k appears once per executed iteration."""
        plan = []
        for k in range(1, self.n_stages + 1):
            if k not in self.each:
                plan.append((k, self.titles.get(k, ""), unparse_stmts(self.stage_stmts(k), self.keep_calls), None))
                continue
            for binding in self._iterations(k, records or []):
                plan.append((k, self._each_title(k, binding), self._each_body(k, binding), binding))
        return plan

    def slices(self, header_word="stage", records=None):
        """Slice texts for one episode (`records` are needed when the program has `[stage each]` loops)."""
        plan = self.stage_plan(records)
        n = len(plan)
        return [f"# [{header_word} {i}/{n}] {title}\n{body}\n" for i, (k, title, body, _) in enumerate(plan, 1)]

    # --- records -> stages ---------------------------------------------------------------------------
    def _stmt_at(self, line):
        """Top-level main statement (of the ANNOTATED program) that owns an executed-program line."""
        src_main = self.exec_main if self.exec_main is not None else self.main
        for i, s in enumerate(src_main):
            if s.lineno <= line <= s.end_lineno:
                return self.main[i]
        return None

    def _helper_at(self, line):
        src_helpers = self.exec_helpers if self.exec_helpers is not None else self.helpers
        for i, fn in enumerate(src_helpers):
            if fn.lineno <= line <= fn.end_lineno:
                return self.helpers[i]
        return None

    def _calls_in(self, node):
        return {n.func.id for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

    def helper_stages(self):
        """helper name -> set of stages whose code reaches it (transitively through other helpers)."""
        by_name = {fn.name: fn for fn in self.helpers}
        out = {}
        for k in range(1, self.n_stages + 1):
            seen, todo = set(), set()
            for s in self.stage_stmts(k):
                todo |= self._calls_in(s)
            while todo:
                name = todo.pop()
                if name in seen or name not in by_name:
                    continue
                seen.add(name)
                out.setdefault(name, set()).add(k)
                todo |= self._calls_in(by_name[name])
        return out

    def stage_of_records(self, records):
        """Monotonic stage-INSTANCE index per record (1-based, indexing stage_plan(records))."""
        hs = self.helper_stages()
        plan = self.stage_plan(records)
        inst_of = {}                            # (k, frozen binding | None) -> instance index
        for i, (k, _, _, b) in enumerate(plan, 1):
            inst_of[(k, tuple(sorted((n, _freeze(v)) for n, v in b.items())) if b else None)] = i

        def instances(k, r):
            if k in self.each:
                loc = r.get("top_locals") or {}
                targets = self._for_targets(self.stage_stmts(k)[0])
                if all(t in loc for t in targets):
                    key = (k, tuple(sorted((t, _freeze(loc[t])) for t in targets)))
                    return {inst_of[key]} if key in inst_of else set()
                return {i for (kk, _), i in inst_of.items() if kk == k}
            return {inst_of[(k, None)]} if (k, None) in inst_of else set()
        cur, out = 1, []
        self.ambiguous = 0
        for r in records:
            ln = int(r.get("top_line") or r["line"])
            st = self._stmt_at(ln)
            if st is not None:
                sec = self.sections[st]
                cands = instances(sec, r) if isinstance(sec, int) else set()
            else:
                fn = self._helper_at(ln)
                cands = set()
                for k in (hs.get(fn.name, ()) if fn else ()):
                    cands |= instances(k, r)
                if len([c for c in cands if c >= cur]) > 1:
                    self.ambiguous += 1
            later = [c for c in cands if c >= cur]
            k = min(later) if later else (max(cands) if cands else cur)
            k = max(k, cur)
            out.append(k)
            cur = k
        return out

    def stage_frames(self, records, n_frames):
        """[(start, end)] per stage INSTANCE (see stage_plan) from the first-record rule; an instance that
        never issued a motion is zero-length where the next one begins (legacy: carry_end = lift_end)."""
        n = len(self.stage_plan(records))
        ks = self.stage_of_records(records)
        starts = {}
        for r, k in zip(records, ks):
            starts.setdefault(k, int(r["frame_start"]))
        a_of = {k: (0 if k == 1 else starts.get(k)) for k in range(1, n + 1)}
        nxt = n_frames
        for k in range(n, 0, -1):
            if a_of[k] is None:
                a_of[k] = nxt
            nxt = a_of[k]
        starts_list = [a_of[k] for k in range(1, n + 1)] + [n_frames]
        return [(starts_list[i], starts_list[i + 1]) for i in range(n)]


# ----------------------------------------------------------------------------- episodes / caches
def kept_episodes(align_dir):
    """Alignment JSONs of successful episodes sorted by seed (the LeRobot converter's order)."""
    align_dir = Path(align_dir)
    man = align_dir / "manifest.jsonl"
    if man.exists():
        rows = [json.loads(l) for l in open(man) if l.strip()]
        seeds = sorted(m["seed"] for m in rows if m.get("success"))
    else:
        seeds = sorted(int(re.search(r"episode(\d+)", p.name).group(1)) for p in align_dir.glob("episode*.json"))
    eps = []
    for sd in seeds:
        al = json.load(open(align_dir / f"episode{sd}.json"))
        if not al.get("success", True):
            continue
        eps.append(dict(seed=sd, num_frames=int(al["num_frames"]), instruction=al.get("instruction"),
                        records=al["records"], code=al.get("code")))
    codes = {e["code"] for e in eps if e["code"]}
    assert len(codes) <= 1, "alignment files embed different program texts"
    return eps


def build(prog, eps, out, header_word="stage", n_train=50, style="sig", only_called=True, setup_names=True):
    import torch
    out = Path(out)
    pre = prog.prompt_preamble(style, only_called, setup_names)
    slices = None if prog.each else prog.slices(header_word)
    ep_dir = out / "slices_ep"
    ep_dir.mkdir(parents=True, exist_ok=True)
    for old in ep_dir.glob("ep_*.json"):
        old.unlink()
    total = sum(e["num_frames"] for e in eps)
    cyc = torch.full((total,), -1, dtype=torch.long)
    off, rows = 0, []
    for ei, e in enumerate(eps):
        fr = prog.stage_frames(e["records"], e["num_frames"])
        for k, (a, b) in enumerate(fr):
            cyc[off + a: off + b] = k
        ep_slices = prog.slices(header_word, e["records"]) if prog.each else slices
        json.dump({"preamble": pre, "slices": ep_slices}, open(ep_dir / f"ep_{ei:06d}.json", "w"))
        rows.append(dict(ep=ei, seed=e["seed"], frames=e["num_frames"], phases=fr,
                         present=[int(b > a) for a, b in fr]))
        off += e["num_frames"]
    assert int(cyc.min()) >= 0
    (out / "preamble.txt").write_text(pre)           # the prompt text (what the loader prepends)
    torch.save({"cycle_by_idx": cyc, "action_size": 14}, out / "cycle_by_idx.pt")
    idx = list(range(len(eps)))
    json.dump(idx[:n_train], open(out / "episodes_train.json", "w"))
    json.dump(idx[n_train:], open(out / "episodes_val.json", "w"))
    json.dump(idx, open(out / "episodes_all.json", "w"))
    json.dump(rows, open(out / "stage_frames.json", "w"), indent=0)
    return rows


def regress(prog, eps, legacy):
    """Byte-compare preamble + slices with the legacy cache and the frame vector with cycle_train_v7s.pt."""
    import torch
    legacy = Path(legacy)
    ref = json.load(open(sorted((legacy / "policy_code_slices_ep_v7s").glob("ep_*.json"))[0]))
    pre, slices = prog.preamble(), prog.slices("step")
    ok = True
    if pre != ref["preamble"]:
        ok = False; print("PREAMBLE DIFFERS"); _diff(ref["preamble"], pre)
    if slices != ref["slices"]:
        ok = False; print("SLICES DIFFER")
        for k, (a, b) in enumerate(zip(ref["slices"], slices), 1):
            if a != b:
                print(f"--- stage {k}"); _diff(a, b)
    cyc_ref = torch.load(legacy / "cycle_train_v7s.pt")["cycle_by_idx"]
    off, bad = 0, 0
    for e in eps:
        fr = prog.stage_frames(e["records"], e["num_frames"])
        got = torch.full((e["num_frames"],), -1, dtype=torch.long)
        for k, (a, b) in enumerate(fr):
            got[a:b] = k
        want = cyc_ref[off: off + e["num_frames"]]
        if not torch.equal(got, want):
            bad += 1
            if bad <= 3:
                print(f"seed {e['seed']}: frames differ; engine {fr}; legacy boundaries "
                      f"{[int(i) for i in (torch.nonzero(want[1:] != want[:-1]).flatten() + 1)]}")
        off += e["num_frames"]
    if off != len(cyc_ref):
        ok = False; print(f"frame total differs: engine {off} vs legacy {len(cyc_ref)}")
    if bad:
        ok = False
    print(f"regress: preamble {'OK' if pre == ref['preamble'] else 'DIFF'}, slices "
          f"{'OK' if slices == ref['slices'] else 'DIFF'}, frames {len(eps) - bad}/{len(eps)} episodes identical")
    return ok


def _diff(a, b):
    import difflib
    for l in difflib.unified_diff(a.splitlines(), b.splitlines(), "legacy", "engine", lineterm="", n=1):
        print("   " + l)


def _resolve(target):
    """`target` is a task name (configs/robotwin/tasks/<task>/) or a program path -> (program path, task dict|None)."""
    if Path(target).is_file():
        return Path(target), None
    t = config.task(target)
    return Path(t["frozen"]), t


def _executed_source(task, explicit):
    if explicit:
        return Path(explicit).read_text()
    if task:
        src = (config.load_task_yaml(task["name"]).get("source") or {}).get("frozen_program")
        if src and Path(src).is_file():
            return Path(src).read_text()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("target", help="task name or program path")
    c.add_argument("--executed", default=None, help="frozen (un-annotated) program: verify comment-only diff "
                                                    "(default for a task: task.yaml source.frozen_program, if present)")
    pr = sub.add_parser("preamble"); pr.add_argument("target", help="task name or program path")
    pr.add_argument("--write", action="store_true", help="write tasks/<task>/preamble.txt (task targets only)")
    for s in (c, pr):
        s.add_argument("--style", choices=["sig", "desc", "full"], default="sig",
                       help="prompt stubs: signature only (default), with return descriptions, or the legacy full preamble")
        s.add_argument("--all-helpers", action="store_true", help="list every helper, not only the ones a stage calls")
        s.add_argument("--no-setup-names", action="store_true", help="omit the `# Setup defines:` line")
    for name in ("build", "regress"):
        s = sub.add_parser(name)
        if name == "build":
            s.add_argument("--style", choices=["sig", "desc", "full"], default="sig",
                           help="prompt stubs: signature only (default), with return descriptions, or the legacy full preamble")
            s.add_argument("--all-helpers", action="store_true", help="list every helper, not only the ones a stage calls")
            s.add_argument("--no-setup-names", action="store_true", help="omit the `# Setup defines:` line")
        s.add_argument("--task", default=None, help="task name under configs/robotwin/tasks/")
        s.add_argument("--program", default=None, help="annotated program path (default: the task's frozen_program.py)")
        s.add_argument("--align-dir", required=True)
        if name == "build":
            s.add_argument("--out", required=True); s.add_argument("--header-word", default="stage")
            s.add_argument("--n-train", type=int, default=50)
        else:
            s.add_argument("--legacy", required=True)
    a = ap.parse_args()
    if a.cmd in ("check", "preamble"):
        path, task = _resolve(a.target)
        executed = _executed_source(task, getattr(a, "executed", None))
        prog = Program(path, executed_src=executed)
        pre = prog.prompt_preamble(a.style, not a.all_helpers, not a.no_setup_names)
        if a.cmd == "preamble":
            if a.write:
                if not task:
                    sys.exit("--write needs a task name")
                Path(task["preamble"]).write_text(pre); print(f"wrote {task['preamble']}")
            else:
                print(pre, end="")
            return 0
        errs = prog.check()
        if task and Path(task["preamble"]).is_file() and Path(task["preamble"]).read_text() not in (pre, prog.preamble()):
            errs.append(f"{task['preamble']} is stale: run `slicing.py preamble {task['name']} --write`")
        for s in prog.main:
            print(f"{s.lineno:4d}-{s.end_lineno:4d}  {str(prog.sections[s]):6s}  {prog.lines[s.lineno - 1].strip()[:70]}")
        print(f"{prog.n_stages} stages: " + " | ".join(f"{k}: {prog.titles.get(k)}" for k in range(1, prog.n_stages + 1)))
        print("executed-program equivalence: " + ("verified" if executed else "not checked (no executed source)"))
        print("OK" if not errs else "ERRORS:\n  " + "\n  ".join(errs))
        return 0 if not errs else 1
    if not (a.task or a.program):
        sys.exit("give --task or --program")
    eps = kept_episodes(a.align_dir)
    prog = Program(a.program or config.task(a.task)["frozen"], executed_src=eps[0]["code"] if eps else None)
    errs = prog.check()
    if errs:
        sys.exit("contract errors:\n  " + "\n  ".join(errs))
    if eps and not any("top_line" in r for r in eps[0]["records"]):
        amb = sum((prog.stage_of_records(e["records"]), prog.ambiguous)[1] for e in eps)
        if amb:
            print(f"WARNING: alignment records carry no `top_line`; {amb} helper records were attributed by "
                  f"heuristic (shared helpers across stages) — re-collect with top_line for trustworthy frames")
    if a.cmd == "build":
        rows = build(prog, eps, a.out, a.header_word, a.n_train, a.style, not a.all_helpers, not a.no_setup_names)
        empty = sum(1 for r in rows for p in r["present"] if not p)
        counts = sorted({len(r["phases"]) for r in rows})
        print(f"built {len(rows)} episodes, {prog.n_stages} stage slots, {counts} stage instances per episode "
              f"-> {a.out} ({empty} empty stage instances)")
        return 0
    return 0 if regress(prog, eps, a.legacy) else 1


if __name__ == "__main__":
    sys.exit(main())
