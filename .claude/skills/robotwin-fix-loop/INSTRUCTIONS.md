---
name: robotwin-fix-loop-instructions
description: End-to-end steps for the RoboTwin fix loop — baseline, Stage-1 diagnosis and program authoring on the debug seeds, one-shot Stage-2 validation on the locked seeds.
---

# RoboTwin fix loop — instructions

Layer 2 of the workflow (`../../docs/ROBOTWIN.md`): an agent reads the LLM baseline's failed trials
for one task, authors one generalizable program, and the coordinator validates it once on held-out seeds.
Read [`../../CLAUDE.md`](../../../codegen/sim/CLAUDE.md) first, then [SKILL.md](SKILL.md).

1. **Environment.** From `codegen/sim`: `source env.sh && codegen_sim_preflight`. Every simulator command
   runs under `$ROBOTWIN_PY`. The privileged configs (`iterate/configs/priv_{clean,randomized}.yaml`)
   need no perception servers; the perception config needs Molmo, SAM3 and GraspNet (`../../CLAUDE.md` § APIs).
2. **Baseline** (optional when baseline trials already exist for the task):

       $ROBOTWIN_PY iterate/scripts/run_robotwin_batch.py \
           --args.base-config-path iterate/configs/priv_clean.yaml \
           --args.tasks <task> --args.resume-idx 101 --args.total-trials 105 \
           --args.server-url <chat-completions URL> --args.models <model> [--args.api-key ...]

   Output tree: `$CODEGEN_SIM_AGENT_ROOT/baseline/<task_config>/<task>/<model>/run/trial_*` with per-seed
   `code.py`, `trace.json`, `summary.txt` and videos.
3. **Stage 1.** The coordinator ([main-agent-prompt.md](main-agent-prompt.md)) dispatches
   [subagent-prompt.md](subagent-prompt.md) once per task. The subagent iterates on all ten debug seeds
   (101-105 `demo_clean`, 151-155 `demo_randomized`) and writes `fix_code.py` + `findings.md` under
   `$CODEGEN_SIM_AGENT_ROOT/fix_loop/<task>/`.
4. **Stage 2** (coordinator only, with the human's explicit go-ahead): the locked seeds are one-shot.
   `GPUS="<ids>" bash iterate/scripts/run_stage2.sh <task>` freezes `fix_code.py` (copy + sha256), checks the stage
   markers, and replays the default locked set (201-205 clean + 251-260 randomized) into
   `$CODEGEN_SIM_VALIDATION_ROOT/stage2/<task>/`. Replaying a locked seed during iteration invalidates the
   benchmark; the randomized debug seeds are the pre-validation robustness check.
4b. **Freeze** (after a PASS): write the score line into `<out_root>/STAGE2_RESULT.md`, then promote the program into
   the handoff the data pipeline reads:

       python3 iterate/scripts/freeze.py <task> --program <out_root>/frozen_program.py --stage2 <out_root>/STAGE2_RESULT.md

   This writes `frozen/<task>/{frozen_program.py, frozen_program.sha256, STAGE2_RESULT.md, task.yaml}`; nothing
   downstream reads the fix-loop or Stage-2 directories.
5. **Per-seed replay tool:**

       CUDA_VISIBLE_DEVICES=<gpu> $ROBOTWIN_PY iterate/scripts/replay_trial.py --code <program.py> --seed <N> \
           --task <task> --config iterate/configs/priv_clean.yaml --output-dir <dir>

   The reward is in the trial directory name (`..._reward_1.000`); the launcher inserts `oracle/` before the last output component:
   `--output-dir <parent>/<run>` produces `<parent>/oracle/<run>/trial_*`.
6. **Read every score in distinct scenes.** Unstable-scene retries collapse seeds onto one scene,
   task-specifically. Run `iterate/scripts/stage2_report.py <out_root>`  and report the scene
   count with the seed count.
7. **Freezing bar.** A program is collection-ready only when it clears both the success gate (Stage 2
   ≥ 12/15) and the trainability audit in [subagent-prompt.md](subagent-prompt.md) § Trainability. Success
   rate alone does not make a program a usable data oracle.
8. **Records.** The pipeline's records are `findings.md` per task, the ledger
   `$CODEGEN_SIM_AGENT_ROOT/progress/robotwin_fix_loop_progress.md` (`iterate/scripts/gen_progress.py`), and the coordinator's host
   report under `$CODEGEN_SIM_AGENT_ROOT/_reports/`. Write to those, not to ad-hoc notes.
