---
name: robotwin-fix-loop
description: "Run the RoboTwin fix loop: one Stage-1 subagent per task authors a program on the debug seeds; the coordinator freezes it and validates once on the locked seeds, then promotes reusable patterns into the skill library."
---

# RoboTwin fix loop

Layer 2 of the nested workflow (multi-turn ⊂ fix loop ⊂ evosearch). Start with
[INSTRUCTIONS.md](INSTRUCTIONS.md).

## Files

| File | Purpose |
|---|---|
| [INSTRUCTIONS.md](INSTRUCTIONS.md) | Entry point: environment, baseline, Stage 1, Stage 2, tools |
| [main-agent-prompt.md](main-agent-prompt.md) | Coordinator runbook: task list, dispatch, GPU ledger, Stage 2, promotion, host report |
| [subagent-prompt.md](subagent-prompt.md) | Per-task Stage-1 prompt template |
| [../skills/](../robotwin-program-authoring/) | Shared skill library (coordinator-owned) |
| [../program-contract.md](../robotwin-program-authoring/program-contract.md) | Stage markers every training program carries |
| [../evosearch/](../robotwin-evosearch/) | Layer 3, for strategy-level failures this loop leaves below the bar |

## Roles

| Role | Runs | Reads | Writes |
|---|---|---|---|
| Coordinator | Stage 2 (`run_stage2.sh`), `gen_progress.py` | `findings.md`; `fix_code.py` only to lift a snippet | `../skills/`, the ledger, the host report |
| Stage-1 subagent | debug replays (`replay_trial.py`) | baseline trials, traces, videos, task source, skills | `fix_code.py`, `findings.md`, `attempts/` |

## Seed discipline

| Set | Seeds | Who may run it |
|---|---|---|
| Debug | 101-105 clean · 151-155 randomized | every Stage-1 round |
| Locked validation | default 201-205 clean · 251-260 randomized | `run_stage2.sh`, once per program |

The 201-270 band is reserved for validation. The evaluator explicitly rejects 201–210 and
251–270; collection rejects starts below 1001 by default. These guards do not cover newly assigned ranges. A task
whose default range was spent by an earlier program mints a fresh range (`CLEAN_SEEDS` / `RAND_SEEDS`) and a
new `out_root`; never re-run a range already reported on. Full policy: `../../CLAUDE.md` § Seed policy.

## Stopping criteria (Stage 1)

- The candidate is stable across a full round on all ten debug seeds, or
- four rounds are used. Deliver the best program either way; a task that plateaus low is triage
  information, not a failure.

## What a finished task looks like

`fix_code.py` and a `findings.md` ending in `## Collection suitability` (the ledger keys on that heading),
a Stage-2 result under `$CODEGEN_SIM_VALIDATION_ROOT/stage2/<task>/`, its score read in distinct scenes, and
the task's generalizable patterns merged into `../skills/`.
