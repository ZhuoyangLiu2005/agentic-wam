---
name: robotwin-fix-loop-main-agent-prompt
description: Coordinator guide for the RoboTwin fix loop — dispatch one Stage-1 subagent per task on a GPU set, run Stage 2 on the freed GPUs, promote skills, write the host report.
---

# RoboTwin fix loop — coordinator guide

> **What:** run the fix loop over the task list you were given, so every task ends with a `fix_code.py`
> (Stage 1) and a held-out Stage-2 score. That score ranks tasks for aligned-data collection.
> **Ledger:** `$CODEGEN_SIM_AGENT_ROOT/progress/robotwin_fix_loop_progress.md`; regenerate with
> `$ROBOTWIN_PY iterate/scripts/gen_progress.py` before every read.
> **Subagent template:** [subagent-prompt.md](subagent-prompt.md)

---

## Scope

- Your task list is the file passed to you: one task name per line, a subset of
  `iterate/configs/tasks_all.txt`. **Never touch a task outside it.** Several coordinators may run the
  same campaign on their own slices against one shared output tree.
- Lifecycle, as the ledger reports it: `pending → stage1-running → stage1-done → done`.
  - `pending` → dispatch a Stage-1 subagent. It iterates on all ten debug seeds (101-105 clean + 151-155
    randomized) and writes `fix_code.py` + `findings.md`.
  - `stage1-done` → **you** run Stage 2 yourself, in the background, on the GPU set your ledger just freed.
    Always pass the set explicitly; with `GPUS` unset the script auto-detects free cards and can grab ones
    a subagent is using:

        GPUS="<freed gpus>" bash iterate/scripts/run_stage2.sh <task>        # CONC=n caps concurrency (default 2 per card)

    It freezes the program, runs the locked default set once, and writes
    `$CODEGEN_SIM_VALIDATION_ROOT/stage2/<task>/`. **Never dispatch a subagent for a `stage1-done` task**:
    it would redo Stage 1 and could overwrite a validated program.
  - `done` → never touch again. Stage 2 is one shot per program; the script refuses a re-run. A later
    program for the same task needs a fresh seed range and a new `out_root` (`../../CLAUDE.md` § Seed policy).
- Subagents never run any seed outside 101-105 / 151-155.

## Loop

1. **Regenerate the ledger** (`$ROBOTWIN_PY iterate/scripts/gen_progress.py`) and read the rows for your tasks only.
2. **Dispatch** one Stage-1 subagent per `pending` task with [subagent-prompt.md](subagent-prompt.md),
   `SUBAGENT_GPUS` filled from your ledger. At most one subagent per GPU set in flight; a subagent runs its
   ten debug replays in parallel across the GPUs you give it, so two subagents must never share a GPU.
3. **Never end your turn while a subagent is running.** When you run headless (`claude -p`) there is no
   next turn to be notified in, and background work is killed shortly after your turn ends, which truncates
   the subagent and leaves a half-written `fix_code.py`. After dispatching, keep the turn alive with a
   blocking wait, run as a normal foreground Bash call:

   ```bash
   # returns when every dispatched task has a findings.md, or after 2 h
   deadline=$((SECONDS + 7200))
   while [ $SECONDS -lt $deadline ]; do
     miss=0
     for t in <the tasks you just dispatched>; do
       [ -f $CODEGEN_SIM_AGENT_ROOT/fix_loop/$t/findings.md ] || miss=$((miss+1))
     done
     [ $miss -eq 0 ] && break
     sleep 120
   done
   echo "wait finished: $miss task(s) still missing findings.md"
   ```

   Dispatch a wave (one subagent per GPU set), wait for the wave, process results, dispatch the next wave.
   Do not poll GPUs or tail logs in between; one blocking wait is enough.
4. **On completion of a wave**, for each task that produced both `fix_code.py` and `findings.md`: start its
   Stage 2 in the background on the GPUs the subagent just freed; read its `findings.md` (only) and promote
   generalizable patterns into `../skills/` per [../skills/README.md](../robotwin-program-authoring/SKILL.md); free its GPUs in
   the ledger when Stage 2 finishes; dispatch the next `pending` task.
5. **Stop** when every task in your slice is `done` or blocked with a written reason. Write the host report
   to `$CODEGEN_SIM_AGENT_ROOT/_reports/<host>_fix_loop.md`: one row per task with
   - the **debug score** (x/10) and the **Stage-2 score** as `randomized x/10, clean x/5`, each with its
     distinct-scene count (`iterate/scripts/stage2_report.py`). The randomized number is the primary ranking key:
     collections are mostly randomized episodes, so randomized yield decides a task's collection value;
   - the subagent's **motion-economy line** (frames/episode and gripper cycles versus `play_once`, redundant
     motions). Between tasks of similar score, the one whose episodes match the expert demo's structure is
     worth more as training data;
   - the **collection-suitability verdict** and the skills admitted;
   - for every task scoring poorly, whether the failures are **STRATEGY-level** (a different approach might
     work → an evosearch candidate) or **CAPABILITY-level** (the API cannot express the task → drop it).
     Escalate to evosearch only for strategy-level near-misses.

## GPU ledger

You own only the usable GPUs on this host (the launcher tells you which). A GPU set is held by a task from
subagent dispatch until that task's Stage 2 finishes:

```
GPU <ids> : SUBAGENT <task_name>  (dispatched <time>)
GPU <ids> : STAGE2   <task_name>  (background, started <time>)
GPU <ids> : FREE
```

One job per GPU set; a job is either a Stage-1 subagent or a Stage-2 run. When a subagent returns, start
that task's Stage 2 on the same GPUs and release them only when it finishes. Never trust `nvidia-smi` to
decide a GPU is free (both job types idle between replays); the ledger is authoritative. Rebuild it from
running processes if you resume a session.

## Rules

- **Never read task-specific debug files** (`trace.json`, `code.py`, `summary.txt`, videos). That is
  subagent work. You read `findings.md`, plus `fix_code.py` only to lift a snippet into a skill file.
- **You are the only writer of `../skills/`.** Subagents write `findings.md`; you promote. Merge into
  existing sections; never append a paraphrase.
- **Never edit** the API (`capsim/integrations/`), the env wrappers, the configs or the replay tooling.
  Problems there go into your host report for a human.
- Skills written by other coordinators appear in the same shared files. Re-read a skill file before editing
  it and keep edits additive.
- A task whose `fix_code.py` exists but whose `findings.md` is missing or incomplete was truncated; the
  ledger reports it as `pending (TRUNCATED)`. Re-dispatch a subagent for it. Never run Stage 2 on it, and if
  Stage 2 was already spent on it, its re-validation must use a fresh seed range.
