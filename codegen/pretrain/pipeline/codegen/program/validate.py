"""Static validation of a generated slice (part 1 of 2 — the other half is the
gripper-EFFECT grounding in `effects.py`)."""
from __future__ import annotations

import ast
import inspect

from codegen.api.active import ARG_VOCAB, EMBODIMENT, _API_SIG
from codegen.library import SKILL_DEFS


def _skill_signature(fndef):
    """inspect.Signature for a composed-skill function (library or program-inline)."""
    n_def = len(fndef.args.defaults)
    params = []
    for i, a in enumerate(fndef.args.args):
        has_default = i >= len(fndef.args.args) - n_def
        params.append(inspect.Parameter(
            a.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None if has_default else inspect.Parameter.empty))
    return inspect.Signature(params)


#: message tail shared by every cut-off-generation problem, so the repair round
#: is told to RE-EMIT the slice rather than to patch the fragment.
_CUTOFF_FIX = ("your output appears to have been CUT OFF mid-generation — re-emit the "
               "WHOLE slice, ending with a complete statement that codes every event "
               "you were asked to reproduce")


def _dangling_comment(code):
    """Trailing-comment fingerprint of a cut-off generation.

    A slice whose LAST non-blank line is a comment, in a slice that does contain
    code, is the signature of a generation that stopped while narrating its next
    action (`# Right arm releases the` — the events that comment describes are
    never written). Slices that are ONLY comments are legal and common (a
    zero-event slice is matched by any contact-free code), hence the call test.

    Measured on the yam v3 corpus (401 slices): 0 false positives in 379 matched
    slices, fires on 13 of the 22 mismatched ones."""
    ne = [ln for ln in code if ln.strip()]
    if not ne or not ne[-1].lstrip().startswith("#"):
        return []
    if not any("(" in ln for ln in ne):
        return []                       # comment-only slice — legitimate
    return [f"line {len(code)}: the slice ends with a dangling comment "
            f"({ne[-1].strip()[:60]!r}) and no code for it — {_CUTOFF_FIX}"]


def validate_code(code, prog_so_far=""):
    """VALIDATION stage, part 1 of 2 (static; the other is the gripper-EFFECT grounding).
    Returns a list of problem strings, empty = valid:

      * the program-so-far + slice must PARSE (a slice may legally continue an earlier
        block, so syntax is judged in program context, with line numbers mapped back to
        the slice);
      * every ``robot.<fn>(...)`` call must name a real function of the active
        embodiment's API and
        pass arguments its signature accepts (catches hallucinated APIs / bad kwargs);
      * a plain-function call to a library (or program-inline) SKILL must pass `robot`
        first and fit the skill's signature;
      * an API function must be called on ``robot`` (a bare ``pick(...)`` is the skill
        calling convention applied to an API verb);
      * the slice must not carry a CUT-OFF-GENERATION fingerprint (dangling trailing
        comment, or a statement that is a bare name / attribute access). These parse
        and call nothing, so without this check they validate clean, reach
        ``program.py``, and cost the slice its entire event budget at the gate."""
    snippet = "\n".join(code)
    full = f"{prog_so_far}\n{snippet}" if prog_so_far else snippet
    base = prog_so_far.count("\n") + 1 if prog_so_far else 0
    cutoff = _dangling_comment(code)     # textual: survives an unparseable snippet
    try:
        ftree = ast.parse(full)
    except SyntaxError as e:
        ln = max((e.lineno or 1) - base, 1)
        return cutoff + [f"SyntaxError: {e.msg} at line {ln} of your code: "
                         f"{(e.text or '').strip()}"]
    skill_defs = dict(SKILL_DEFS)   # library skills (imported in the header)
    skill_defs.update({n.name: n for n in ftree.body if isinstance(n, ast.FunctionDef)})
    probs = list(cutoff)
    try:
        nodes, off = list(ast.walk(ast.parse(snippet))), 0
    except SyntaxError:
        # CONTINUATION slice: parses only in program context (an indented block
        # body, an `else` branch). These used to skip every per-call check —
        # the zero-validation path — so hallucinated API names, bad kwargs and
        # out-of-vocabulary literals survived exactly in the structurally
        # trickiest slices. Instead, take the slice's OWN nodes out of the
        # program-context tree by line range and run the SAME checks, with
        # line numbers mapped back to the slice.
        nodes, off = [n for n in ast.walk(ftree)
                      if getattr(n, "lineno", 0) > base], base
    # cut-off fingerprint, AST form: a statement that is a bare name (`robot`,
    # `nudge_in_`) or a dangling attribute access (`robot.open_`). Both parse and
    # call nothing, so every other check passes them through.
    for node in nodes:
        if isinstance(node, ast.Expr) and isinstance(node.value, (ast.Name, ast.Attribute)):
            frag = ast.unparse(node.value) if hasattr(ast, "unparse") else "<fragment>"
            probs.append(f"line {node.lineno - off}: `{frag}` is not a call — it calls no API "
                         f"function and grounds to nothing; {_CUTOFF_FIX}")
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "robot"):
            name = node.func.attr
            sig = _API_SIG.get(name)
            if sig is None:
                hint = (f" (did you mean the skill {name}(robot, ...)?)"
                        if name in skill_defs else "")
                probs.append(f"line {node.lineno - off}: robot.{name}(...) is not a function of "
                             f"the {EMBODIMENT} API — use only the documented API{hint}")
                continue
            label = f"robot.{name}"
        elif isinstance(node.func, ast.Name) and node.func.id in skill_defs:
            name = node.func.id
            sig = _skill_signature(skill_defs[name])
            if not (node.args and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "robot"):
                probs.append(f"line {node.lineno - off}: skill {name}(...) must be called "
                             f"with `robot` as its first argument")
                continue
            label = name
        elif isinstance(node.func, ast.Name) and node.func.id in _API_SIG:
            # An API verb called with the SKILL convention (`regrasp('tape', 'left')`).
            # The grounder resolves it either way (codegen.program.calls._callee_name
            # keys on the name), so reconciliation is unaffected — but program.py is
            # the deliverable and the bare form would NameError if it were ever run.
            # 18 such lines in one v3 task, all `regrasp`.
            probs.append(f"line {node.lineno - off}: {node.func.id}(...) is an API function — "
                         f"call it as robot.{node.func.id}(...); the bare "
                         f"name(robot, ...) form is only for library skills")
            continue
        else:
            continue
        try:
            sig.bind(*[None] * len(node.args),
                     **{kw.arg: None for kw in node.keywords if kw.arg})
        except TypeError as e:
            probs.append(f"line {node.lineno - off}: {label}(...) — {e}")
            continue
        # literal ARGUMENT VOCABULARIES (sides, and for ego the direction/
        # relation/axis enums — codegen.api.active.ARG_VOCAB): an out-of-
        # vocabulary literal grounds to ZERO events or silently garbles the
        # motion semantics, so catch it here where the repair loop can fix it.
        lit = lambda a: a.value if isinstance(a, ast.Constant) else None  # noqa: E731
        try:
            bound = dict(sig.bind_partial(
                *[lit(a) for a in node.args],
                **{kw.arg: lit(kw.value) for kw in node.keywords if kw.arg}).arguments)
        except TypeError:
            bound = {}
        for p, v in bound.items():
            allowed = ARG_VOCAB.get(p)
            if allowed and isinstance(v, str) and v not in allowed:
                probs.append(f"line {node.lineno - off}: {label}(...) — {p}={v!r} is not a valid "
                             f"value; use one of: {', '.join(allowed)}")
    return probs
