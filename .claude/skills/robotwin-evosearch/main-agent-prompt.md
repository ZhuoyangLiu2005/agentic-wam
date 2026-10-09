---
name: robotwin-evosearch-main-agent-prompt
description: Coordinator guide for running evolutionary search on RoboTwin tasks the fix loop left below the bar. Subagents iterate on debug seeds 101-105 + 151-155; each runs its own held-out validation after convergence.
---

# RoboTwin evolutionary search — coordinator guide

> **What:** run a K-candidate × N-iteration search on each assigned task, then validate the winner once on
> the locked seeds.
> **Why:** the fix loop is a single line of reasoning refined in place. When a task fails for a *structural*
> reason, refining the same approach cannot find the fix; searching several distinct approaches in parallel
> can.
> **Subagent template:** [subagent-prompt.md](subagent-prompt.md)
> **Seed rules:** [SKILL.md](SKILL.md) § Seed discipline; read before dispatching.

---

## Preflight

From `codegen/sim`, after `source env.sh && codegen_sim_preflight`, check for every task in your list:

```bash
for t in $(cat <your task list>); do
  echo "== $t"
  ls $CODEGEN_SIM_AGENT_ROOT/fix_loop/$t/{fix_code.py,findings.md}            # Stage-1 output
  ls -d $CODEGEN_SIM_VALIDATION_ROOT/stage2/$t/oracle                           # fix-loop Stage-2 result
  ls -d $CODEGEN_SIM_VALIDATION_ROOT/evosearch_stage2/$t/oracle 2>/dev/null && echo "  evosearch seeds already SPENT"
done
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

A task missing the fix-loop program, findings or completed validation belongs in the fix loop.
Inspect all prior validation reports and assign unused `CLEAN_SEEDS`, `RAND_SEEDS` and `STAGE2_OUT`
to every subagent, including the first evosearch run. Directory checks cannot establish seed freshness.
Replay needs no separate baseline LLM endpoint; candidate generation still uses Claude Code.

---

## Scope

- Your task list is the file passed to you. **Never touch a task outside it.**
- Search only STRATEGY-level failures. Before dispatching a task, confirm from its `findings.md` and from
  `capsim/third_party/RoboTwin/envs/<task>.py` that a different approach is plausible. If the write-up says
  the API cannot express the task, or the scene is physically infeasible (an object that does not fit its
  target), do not dispatch; record it as dropped and move on.
- One task = one subagent = one GPU group. A subagent runs its own iterations **and** its own Stage 2 before
  returning.

---

## Directory layout

```
$CODEGEN_SIM_AGENT_ROOT/evosearch/<task>/
  task_analysis.md                     living analysis, updated every iteration
  run01/
    iter_00/
      candidate_A.py … candidate_F.py
      candidate_A/oracle/eval/trial_*         one trial dir per seed (reward in the name)
      candidate_A/eval/logs/<tag>_<seed>.log
      iter_summary.json                the leaderboard
    iter_01/ …
  evosearch_best_code.py               the winner (written by the subagent)
  findings.md                          subagent summary + Stage-2 result
  BLOCKED                              written instead, if nothing worked

$CODEGEN_SIM_VALIDATION_ROOT/evosearch_stage2/<task>/
  frozen_program.py + .sha256
  oracle/{clean,randomized}/trial_*    locked-seed validation
```

---

## The loop

1. **Run the preflight** above; read the rows for your tasks only.
2. **Dispatch** one subagent per free GPU group with [subagent-prompt.md](subagent-prompt.md), `GPUS`
   filled from your ledger, plus the assigned unused seed ranges and output root.
3. **Never end your turn while a subagent is running.** When you run headless (`claude -p`) there is no next
   turn to be notified in, and background work is killed shortly after a turn ends, which truncates the
   subagent and leaves a half-written program. Keep the turn alive with a blocking wait, run as a normal
   foreground Bash call:

   ```bash
   deadline=$((SECONDS + 14400))     # evosearch is longer than the fix loop
   while [ $SECONDS -lt $deadline ]; do
     miss=0
     for t in <the tasks you just dispatched>; do
       [ -f $CODEGEN_SIM_AGENT_ROOT/evosearch/$t/findings.md ] || \
       [ -f $CODEGEN_SIM_AGENT_ROOT/evosearch/$t/BLOCKED ] || miss=$((miss+1))
     done
     [ $miss -eq 0 ] && break
     sleep 300
   done
   echo "wave done: $miss task(s) still unfinished"
   ```

   Dispatch a wave, wait for the wave, process it, dispatch the next.
4. **On completion of a wave**, for each task: read `findings.md` only, promote generalizable patterns into
   `../skills/` per [../skills/README.md](../robotwin-program-authoring/SKILL.md), free its GPUs, dispatch the next task.
5. **Stop** when every task is finished or BLOCKED. Write
   `$CODEGEN_SIM_AGENT_ROOT/_reports/<host>_evosearch.md`: per task, the fix-loop held-out score, the
   evosearch held-out score (both in seeds and distinct scenes), the winning strategy in one line,
   iterations used, and whether the task now clears the collection bar (12/15).

---

## GPU ledger

```
GPU <ids> : SUBAGENT <task_name>  (dispatched <time>)
GPU <ids> : STAGE2   <task_name>  (background, started <time>)
GPU <ids> : FREE
```

One job per GPU set. Hold the set from dispatch until that task's Stage 2 finishes. Never trust `nvidia-smi`
to decide a GPU is free; both job types idle between replays. Rebuild the ledger from running processes if
you resume.

---

## Rules

- **Never run iterations yourself.** You dispatch, wait, read `findings.md`, promote skills.
- **Never read a task's candidates, traces or videos.** That is subagent work and it will fill your context.
- **You are the only writer of `../skills/`.** Merge into existing sections; never append a paraphrase of
  something already there.
- **Never edit** the API (`capsim/integrations/`), env wrappers, configs, or the replay tooling. Problems
  there go in your report for a human.
- **Never run locked seeds during a search.** `evosearch_eval.py` refuses them; do not work around it.
- **Never re-run a completed Stage 2.** It is one shot per program. A later program needs a fresh seed range
  and a new output root.
