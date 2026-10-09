---
name: robotwin-evosearch-instructions
description: End-to-end entrypoint for RoboTwin Fix Loop + Evolutionary Search — preflight, task eligibility, coordinator and subagent dispatch, held-out validation.
---

# RoboTwin evolutionary search — instructions

Layer 3 of the workflow. Run it only on tasks the fix loop left below the bar **and** classified as
STRATEGY-level (a different approach plausibly works). Never on CAPABILITY-level failures (the API cannot
express the task, or the scene makes it physically impossible); those are dropped, not searched.

1. Complete the setup in [`../../CLAUDE.md`](../../../codegen/sim/CLAUDE.md): `source env.sh && codegen_sim_preflight`
   from `codegen/sim`. Replay needs no separate baseline LLM endpoint; candidate generation still uses Claude Code.
2. Confirm the fix loop finished for each target task: `$CODEGEN_SIM_AGENT_ROOT/fix_loop/<task>/fix_code.py`
   and `findings.md` exist, and `$CODEGEN_SIM_VALIDATION_ROOT/stage2/<task>/oracle/` holds a completed
   held-out score. The search starts from that program and is measured against that number; a task without
   them belongs in the fix loop, not here.
3. Read each target task's `findings.md` and the task source
   `capsim/third_party/RoboTwin/envs/<task>.py` (`check_success()` and `play_once()`). Zero-scoring tasks
   usually turn out to be missing a structural step that `play_once` makes obvious.
4. Review all previous validation reports, including fix-loop validation. Assign unused
   `CLEAN_SEEDS` / `RAND_SEEDS` and a new `STAGE2_OUT` before dispatch; see
   [clean-task-slate.md](clean-task-slate.md). Directory absence alone does not prove seeds are unused.
5. Start a persistent coordinator session, e.g. `screen -S rt-evosearch` or `tmux new -s rt-evosearch`.
6. In that session read [`../../CLAUDE.md`](../../../codegen/sim/CLAUDE.md), this file and [SKILL.md](SKILL.md).
7. Follow [main-agent-prompt.md](main-agent-prompt.md) to pick tasks and dispatch subagents.
8. Fill [subagent-prompt.md](subagent-prompt.md) once per task; point subagents at [../skills/](../robotwin-program-authoring/)
   for the shared skill library.
9. Candidate evaluation runs through `iterate/scripts/evosearch_eval.py` (K candidates × the 10 debug seeds).
10. Held-out validation reuses `iterate/scripts/run_stage2.sh` with an evosearch-specific output root; see
    [SKILL.md](SKILL.md) § Seed discipline.

Use [clean-task-slate.md](clean-task-slate.md) before rerunning a task.
