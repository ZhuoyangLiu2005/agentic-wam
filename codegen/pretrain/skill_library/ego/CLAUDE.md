# ego — callable-skill library rules

This directory is the callable-skill library for the ego embodiment
(egocentric video of two human hands). Code-generation runs READ `skills/`: each
skill's source is shown in the coder prompt and generated programs import and
call skills as plain functions. Treat the library as a read-only snapshot while
any generation run is using it.

## Two kinds of knowledge (keep them apart)

1. **Callable skills** — `skills/<kebab-name>/`: composed Python functions over
   the frozen `EgoHandApi` (`codegen/api/ego.py`). These are VOCABULARY:
   injected into every coder prompt (prompt cost scales with library size)
   and expanded by the grounder, so their contact contract is
   machine-checked. High admission bar.
2. **Pattern notes** — reusable diagnostic or coding knowledge that does NOT
   merit a callable function (detector quirks, prompt phrasings that work,
   altitude conventions, failure signatures). Keep these as prose notes outside
   `skills/`; they are never injected into coder prompts.

## Callable-skill format

One directory per skill under `skills/`:

- `skills/<kebab-name>/skill.py` — ONE composed function whose name equals the
  directory name (kebab -> underscores), first param `robot`, body = core
  `EgoHandApi` calls (pick/grasp/place/release/regrasp/insert/attach/
  handover/rotate/press, close_hand/open_hand, move_arm/goto_relative —
  semantic args, no coordinates) plus comments. STRAIGHT-LINE: no imports,
  no loops, no branches, no nested skill calls — one expansion level keeps
  contact grounding exact. The loader enforces all of this; an invalid
  skill.py is silently dropped, so an unvalidated write is a no-op. The
  docstring is part of the prompt: keep it short, neutral and useful.
- `skills/<kebab-name>/SKILL.md` — YAML frontmatter (`name`, `description`),
  the call signature, when to use / when NOT to use, the contact contract (the
  body's hand close/open sequence), and any generic known limits.

## Validating a skill

```bash
CODEGEN_EMBODIMENT=ego python scripts/check_skill.py <kebab-name>
```

It re-runs loader validation and derives the skill's contact contract by
expansion, exactly as the grounder will. Contact-transparent lines (perception,
free-space motion) never change the contract; the motion scaffold is
semantically plausible but should be treated as proposed until executed.

## Admission bar for callable skills

- **Generic** — the maneuver transfers beyond a single demonstration.
  Situation-specific values (object/tool names, counts, distances, angles) are
  lifted into parameters, with an example value documented in the docstring,
  never baked into the body. A skill whose identity involves a specific tool is
  fine — the tool name is an argument.
- **Recurrent** — the maneuver appears across several distinct tasks, realized
  by a consistent call sequence.
- **Right altitude** — one annotated sub-task; no whole-task orchestration
  recipes.
- **Non-duplicate** — check existing frontmatter first.
- **Valid** — `check_skill.py` passes.

## Retirement

Retire skills that are wrong, duplicated, superseded, or never called. Move
them out of `skills/` (e.g. into an `_attic/` directory) rather than deleting
them, and record why. There is no library size cap, but every skill costs
prompt tokens. Changing the library changes the prompt version — see
`FROZEN.md`.
