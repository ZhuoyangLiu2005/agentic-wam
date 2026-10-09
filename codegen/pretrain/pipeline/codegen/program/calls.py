"""Code-side AST analysis: the effective API calls of a slice in EXECUTION
order, with composed-skill calls (library imports OR program-inline defs)
expanded one level at the call site."""
from __future__ import annotations

import ast

from codegen.api.active import _API_SIG
from codegen.library import SKILL_DEFS


def _callee_name(call):
    f = call.func
    if isinstance(f, ast.Attribute):   # robot.pick(...)
        return f.attr
    if isinstance(f, ast.Name):        # pick(...)
        return f.id
    return None


def _bound_args(name, call, env=None):
    """Map a call's LITERAL args to parameter names via the API signature (so the arm
    argument can be found whether it was passed positionally or by keyword). ``env``
    additionally resolves bare NAME args — used when expanding a skill body, where the
    arm usually arrives through the skill's formal parameter."""
    sig = _API_SIG.get(name)
    if sig is None:
        return {}
    env = env or {}

    def lit(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            return env.get(n.id)
        return None

    try:
        bound = sig.bind_partial(
            *[lit(a) for a in call.args],
            **{kw.arg: lit(kw.value) for kw in call.keywords if kw.arg},
        )
        return dict(bound.arguments)
    except TypeError:
        return {}


class _ProgramCalls(ast.NodeVisitor):
    """Call nodes in program EXECUTION order — does not descend into function
    DEFINITIONS (an inline skill's body contributes at its CALL sites, not where
    it is defined), and UNROLLS `for _ in range(<literal>)` loops so a loop of
    gripper calls implies its true repeated event count (a literal loop that
    grounded once would silently drop every repeat's events)."""

    def __init__(self):
        self.calls = []

    def visit_FunctionDef(self, node):   # noqa: N802
        pass

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):          # noqa: N802
        self.calls.append(node)
        self.generic_visit(node)

    def visit_For(self, node):           # noqa: N802
        it = node.iter
        n = None
        if (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                and it.func.id == "range" and it.args
                and all(isinstance(a, ast.Constant) and isinstance(a.value, int)
                        for a in it.args)):
            try:
                n = len(range(*[a.value for a in it.args]))
            except (TypeError, ValueError):
                n = None
        # unknown trip count -> walk the body ONCE (the old behavior: an under-count
        # is diagnosable at reconcile time, a guess would fabricate events)
        for _ in range(n if n is not None else 1):
            for stmt in node.body:
                self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)


def _skill_env(fndef, call):
    """Literal bindings of a skill call's args to the skill's formal parameters
    (defaults filled first), for expanding the body's effects at the call site."""
    lit = lambda n: n.value if isinstance(n, ast.Constant) else None  # noqa: E731
    formals = [a.arg for a in fndef.args.args]
    env = {}
    for formal, default in zip(formals[len(formals) - len(fndef.args.defaults):],
                               fndef.args.defaults):
        env[formal] = lit(default)
    for i, a in enumerate(call.args):
        if i < len(formals):
            env[formals[i]] = lit(a)
    for kw in call.keywords:
        if kw.arg in formals:
            env[kw.arg] = lit(kw.value)
    return env


def effective_api_calls(code):
    """Every effective API call of the slice in execution order, with skill calls
    (library imports OR program-inline defs) EXPANDED one level at the call site,
    arguments substituted: [(line0, col, seq, api_name, bound_args)]. Returns
    ``None`` if the code does not parse. Nothing is hand-declared — a skill's
    gripper effects are exactly those of its expanded body, in body order,
    attributed to the CALL SITE line."""
    try:
        tree = ast.parse("\n".join(code))
    except SyntaxError:
        return None
    skill_defs = dict(SKILL_DEFS)
    skill_defs.update({n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)})
    tv = _ProgramCalls()
    tv.visit(tree)
    out = []
    for node in tv.calls:
        name = _callee_name(node)
        if isinstance(node.func, ast.Name) and name in skill_defs:
            fndef, env = skill_defs[name], _skill_env(skill_defs[name], node)
            body = sorted((n for n in ast.walk(fndef)
                           if isinstance(n, ast.Call) and _callee_name(n) in _API_SIG),
                          key=lambda n: (n.lineno, n.col_offset))
            out += [(node.lineno - 1, node.col_offset, j,
                     _callee_name(sub), _bound_args(_callee_name(sub), sub, env))
                    for j, sub in enumerate(body)]
        elif name in _API_SIG:
            out.append((node.lineno - 1, node.col_offset, 0, name, _bound_args(name, node)))
    # NO sort: the visitor yields calls in EXECUTION order, which source-position
    # sorting would destroy for unrolled loops (iteration 2 of line 5 runs after
    # iteration 1 of line 6, not before it).
    return out
