---
name: robotwin-evosearch
description: "Run the RoboTwin Fix Loop + Evolutionary Search experiment: multi-candidate iterative search for tasks the fix loop left below the bar, then one-shot held-out validation."
---

# RoboTwin fix loop + evolutionary search

Layer 3 of the nested workflow (multi-turn ⊂ fix loop ⊂ evosearch). Start with
[INSTRUCTIONS.md](INSTRUCTIONS.md).

## Run order

1. [INSTRUCTIONS.md](INSTRUCTIONS.md).
2. Confirm the fix loop produced `fix_code.py` + `findings.md` and a Stage-2 score for the target tasks, and
   that each is classified STRATEGY-level.
3. Follow [main-agent-prompt.md](main-agent-prompt.md) as the coordinator.
4. Fill [subagent-prompt.md](subagent-prompt.md) once per task and dispatch one subagent per GPU group.
5. Read [../skills/](../robotwin-program-authoring/) before writing candidates: the shared library is the accumulated fix-loop
   knowledge, and evosearch should start above it, not rediscover it.
6. Use [clean-task-slate.md](clean-task-slate.md) before any rerun.

## Files

| File | Purpose |
|---|---|
| [INSTRUCTIONS.md](INSTRUCTIONS.md) | Entry point |
| [main-agent-prompt.md](main-agent-prompt.md) | Coordinator runbook |
| [subagent-prompt.md](subagent-prompt.md) | Per-task search prompt |
| [clean-task-slate.md](clean-task-slate.md) | Reset checklist |
| [../skills/](../robotwin-program-authoring/) | Shared RoboTwin skill library |
| [../fix-loop/](../robotwin-fix-loop/) | Layer 2, which produces this layer's starting program |
| `iterate/scripts/evosearch_eval.py` | K candidates × 10 debug seeds → `iter_summary.json` leaderboard |
| `iterate/scripts/run_stage2.sh` | freeze + one-shot locked validation |

## Seed discipline

Debug seeds are 101–105 clean and 151–155 randomized. Use only those during search.
The default Stage-2 set is 201–205 clean and 251–260 randomized. Once a task's range has
been used, a later program needs an unused range and a new output root, including the
first evosearch program after fix-loop validation. A new directory does not make seeds unused.
Review all prior validation reports before assigning `CLEAN_SEEDS`, `RAND_SEEDS` and `STAGE2_OUT`.

```bash
CLEAN_SEEDS="$CLEAN_SEEDS" RAND_SEEDS="$RAND_SEEDS" GPUS="$GPUS" \
  bash iterate/scripts/run_stage2.sh "$TASK" \
  "$CODEGEN_SIM_AGENT_ROOT/evosearch/$TASK/evosearch_best_code.py" "$STAGE2_OUT"
```

Use five clean and ten randomized seeds to retain the default 12/15 collection criterion.
Record the actual ranges and completed-trial counts with both scores; different ranges are
not a paired comparison of identical scenes. Requested seeds can also collapse onto one scene.

Implementation guards are narrower than the workflow rule: `evosearch_eval.py` rejects
201–210 and 251–270; it does not know newly assigned validation ranges. `collect_aligned.sh`
rejects starts below 1001 unless explicitly overridden. Neither guard replaces seed bookkeeping.
See [../../CLAUDE.md](../../../codegen/sim/CLAUDE.md) for the project policy.

## Stopping criteria

A subagent stops searching when any of these holds:

- best candidate ≥ **8/10** on the debug seeds → **solved**
- **5 iterations** completed → **max iterations**
- every candidate family tried has failed at the same physical step, with trace evidence that the API cannot
  express the missing step → **BLOCKED** (write the evidence; do not run Stage 2)

Do not stop merely because an iteration produced a small gain. A plateau means the current family of
approaches is wrong, not that the task is unsolvable; spend the remaining budget on structurally different
approaches.

## What success looks like

The bar is the fix-loop program's held-out score for the same task. If the evosearch program does not beat
the fix-loop program on the debug seeds, the subagent falls back to `fix_code.py` rather than validating
something worse.
