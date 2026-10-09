---
name: robotwin-evosearch-clean-task-slate
description: Checklist before re-searching a RoboTwin task — keep the fix-loop baseline, archive the previous search, mint a fresh seed range if the locked seeds are spent.
---

# Clean task slate — RoboTwin evolutionary search

Run through this before re-searching a task. Archive, never delete; a previous run's hypotheses are the most
useful input the next run has.

1. **The fix-loop baseline must still be there.** `fix_code.py` and `findings.md` under
   `$CODEGEN_SIM_AGENT_ROOT/fix_loop/<task>/`, plus its Stage-2 score under
   `$CODEGEN_SIM_VALIDATION_ROOT/stage2/<task>/`. That score is the bar; without it you cannot tell whether
   the search helped.
2. **Decide what the previous search was.** Candidates, `task_analysis.md` and `iter_summary.json` under
   `$CODEGEN_SIM_AGENT_ROOT/evosearch/<task>/` are either a record to keep or work to supersede. Move them
   to `$CODEGEN_SIM_AGENT_ROOT/evosearch/<task>/_attic_<date>/`.
3. **Review every previous validation range**, including the fix-loop range. Every later program
   needs unused `CLEAN_SEEDS` / `RAND_SEEDS` and a new output root. Record both in the assignment
   and report; a new directory alone does not prove freshness. For an interrupted validation of
   the same frozen bytes, inspect trial completeness before using `RESUME=1`: existing trial
   directories are skipped even if incomplete. Never resume with an edited program.
4. **Check GPU ownership**, not just GPU idleness. Both search replays and Stage 2 idle between trials; the
   coordinator's ledger is authoritative.
5. **Keep search and validation separate.** Nothing under `$CODEGEN_SIM_AGENT_ROOT/evosearch/<task>/run*/`
   may contain a locked-seed trial. If one is there, the run is contaminated: archive it and start over.
6. **Do not promote candidate-specific tricks into `../skills/`.** A pattern earns a place in the shared
   library only if it is phrased for any task and survived on more than the seed that produced it.
