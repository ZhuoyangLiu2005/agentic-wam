---
name: robotwin-evosearch-subagent-prompt
description: Self-contained prompt template for a RoboTwin evolutionary-search subagent. Searches K candidates over the debug seeds until convergence, then runs its own locked-seed validation before returning.
---

# RoboTwin evolutionary search — subagent template

Fill the variables and pass the fenced block as `prompt` to a background subagent. Use a high-capability
model: this needs trace diagnosis across many candidates and hypothesis refinement across iterations.

```python
Agent(
    description="evosearch: <task> GPU<a,b>",
    subagent_type="general-purpose",
    model="opus",
    prompt=<the filled block below>,
    run_in_background=True,
)
```

---

```
## Assignment

TASK:        <robotwin task name>
GPUS:        <space-separated assigned GPU indices>  # yours alone; never touch another GPU
FIXLOOP_S2:  <passed>/<total>     # the fix loop's held-out score — the bar to beat
RUN_DIR:     $CODEGEN_SIM_AGENT_ROOT/evosearch/<task>/run01
CLEAN_SEEDS: <five unused clean seeds, space-separated>
RAND_SEEDS:  <ten unused randomized seeds, space-separated>
STAGE2_OUT:  <new coordinator-assigned validation output directory>

Working directory: the codegen/sim root (`source env.sh` first)
Python:            $ROBOTWIN_PY (from env.sh) for every simulator command

---

## ⛔ Seed lockout

Debug seeds — the ONLY seeds you may run during the search:

    clean       101 102 103 104 105
    randomized  151 152 153 154 155

The validation band 201-270 is off limits during iterations. You run the locked set exactly once, at the
very end, through run_stage2.sh — never through evosearch_eval.py, and never to choose between candidates.
Its hard-coded guard rejects 201–210 and 251–270, not newly assigned ranges.
The debug-only rule still applies to every iteration; do not work around it. Choosing a candidate by its
locked-seed score invalidates the number you then report.

---

## What you are

An evolutionary-search subagent for ONE RoboTwin task. You:

1. search K candidate programs per iteration over the 10 debug seeds,
2. save the winner to `$CODEGEN_SIM_AGENT_ROOT/evosearch/$TASK/evosearch_best_code.py`,
3. run its locked-seed validation yourself,
4. return a structured report.

You have full tool access. The constitution (`CLAUDE.md`: forbidden access, run time versus debug time,
runtime rules) applies to every candidate you write.

---

## Step 1 — Understand why the fix loop failed

Read, in this order:

```bash
cat $CODEGEN_SIM_AGENT_ROOT/fix_loop/$TASK/findings.md          # what was already tried
cat capsim/third_party/RoboTwin/envs/$TASK.py                   # check_success + play_once
ls  $CODEGEN_SIM_VALIDATION_ROOT/stage2/$TASK/oracle/*/         # which seeds failed
cat ../../.claude/skills/robotwin-program-authoring/*.md                                         # the shared library
```

`capsim/third_party/RoboTwin/envs/$TASK.py` is the highest-value read and the one most often skipped. Two
functions decide everything:

- **`check_success()`** — the exact success condition. It frequently constrains objects the program never
  thought to move; several tasks require the *container* to be held off the table, not just the payload
  placed into it.
- **`play_once()`** — RoboTwin's own expert demo: a worked solution. If your starting program is
  structurally different from `play_once`, that difference is usually the entire bug.

Do not treat the previous write-up's conclusions as facts. Verify against the source; "the checker must be
broken" is a conclusion a fix-loop subagent reaches naturally and it is almost always wrong.

Write `$CODEGEN_SIM_AGENT_ROOT/evosearch/$TASK/task_analysis.md`:

```markdown
# Task analysis: <task>

## 1. Success condition           # quote check_success; list EVERY conjunct
## 2. What play_once does         # the upstream expert's structure, step by step
## 3. Where the fix-loop program diverges from it
## 4. Scene geometry              # object sizes, reachability, arm assignment
## 5. Hypotheses for iter_00      # one line per candidate
## 6. Iteration log               # appended each iteration
```

---

## Step 2 — Write K=6 candidates for iter_00

```bash
mkdir -p $RUN_DIR/iter_00
```

- **candidate_A is `$CODEGEN_SIM_AGENT_ROOT/fix_loop/$TASK/fix_code.py` verbatim.** It is your performance
  floor and the thing every other candidate must beat.
- The other five each test a **distinct structural hypothesis**. No two may fail at the same step for the
  same reason. Varying a constant is not a hypothesis; changing which arm holds what, whether a container is
  lifted, the order of sub-goals, or how an object is localized, is.
- Each file starts with a docstring:

  ```python
  """
  Hypothesis:          <what this changes and why it should fix the failure>
  Differs from prior:  <vs candidate_A / vs last iteration's winner>
  Expected failure:    <what you should see in the trace if the hypothesis is wrong>
  """
  ```

  The "expected failure" line is what makes a losing candidate informative instead of merely lost.

**Every candidate carries stage markers** (`# [setup]`, `# [stage k/N] <title>`, `# [stage each] <title>`,
`# [omit]` — [../program-contract.md](../robotwin-program-authoring/program-contract.md)). Structural hypotheses that change the phase
order or the object order must move the markers with them; the winner is frozen only if
`python3 $SLICER check` passes.

**Write for the locked seeds, not the debug seeds.** Ten requested seeds are a small sample and may represent fewer distinct scenes. Prefer strategies that
work for mechanistic reasons (correct object identity, robust localization, physically sound grasp geometry)
over anything tuned to the ten you can see. No hard-coded scene coordinates, no per-seed branches, no
thresholds fitted to observed debug failures: they do not transfer, and a program carrying them is
disqualified from data collection even if it scores well.

---

## Step 3 — Evaluate the iteration

```bash
$ROBOTWIN_PY iterate/scripts/evosearch_eval.py \
    --iter-dir $RUN_DIR/iter_00 --task $TASK --gpus $GPUS        # --per-gpu 2 by default; --record-video for the final pick
```

Run it with `run_in_background=True` and poll; K×10 replays take a while. It prints a leaderboard and writes
`iter_summary.json`. Output format (placeholders, not measured results):

```text
candidate       rate         clean            rand             err            failed seeds
<candidate_id>  <success_%>  <passed>/<total>  <passed>/<total>  <error_count>  <failed_seed_ids>
```

Read the actual values from the current iteration's output; do not infer a candidate's score or ranking
from this format example.

Add `--resume` to continue an interrupted evaluation; pairs with existing trial directories are skipped. Check that those trials actually completed first.

Then select a failed candidate and seed from the current leaderboard and inspect their trial and log:
Replace the placeholders below with that iteration, candidate, setting (`clean` or `randomized`), and seed.

```bash
ls $RUN_DIR/<iteration>/<candidate_id>/oracle/eval/  # per-seed trial dirs (reward in the name)
cat $RUN_DIR/<iteration>/<candidate_id>/eval/logs/<setting>_<seed>.log  # traceback / program stdout
```

---

## Step 4 — Iterate

**Stop when any of these holds:**

- best candidate **≥ 8/10** on the debug seeds → SOLVED
- **5 iterations** completed → MAX_ITERATIONS
- every family tried fails at the same physical step, with trace evidence that no available API can express
  the missing step → BLOCKED

A plateau is not a stopping criterion. If gains have stalled, the current family of approaches is wrong;
spend the remaining budget on a structurally different one, not on tuning the leader.

Per iteration:

1. append `### iter_NN` to `task_analysis.md` §6: leaderboard, which hypotheses are now eliminated and why,
   what is still open
2. write K=6 new candidates seeded from the top 3 survivors, plus at least one deliberately different approach
3. evaluate with the same command against the new iter dir
4. keep the seed set fixed across iterations so the comparison stays fair

---

## Step 5 — Save the winner, validate, report

**5a — pick and save.** The winner must beat candidate_A (the fix-loop program) on the debug seeds. If
nothing does, save candidate_A: validating something worse than what we already have is wasted locked seeds.

```bash
mkdir -p $CODEGEN_SIM_AGENT_ROOT/evosearch/$TASK
cp $RUN_DIR/iter_NN/candidate_X.py $CODEGEN_SIM_AGENT_ROOT/evosearch/$TASK/evosearch_best_code.py
```

If BLOCKED, write the evidence and stop; do not run Stage 2:

```bash
cat > $CODEGEN_SIM_AGENT_ROOT/evosearch/$TASK/BLOCKED << 'EOF'
<which families were tried, the step they all failed at, and the trace evidence
that the API cannot express it>
EOF
```

**5b — locked-seed validation, ONE SHOT:**

```bash
CLEAN_SEEDS="$CLEAN_SEEDS" RAND_SEEDS="$RAND_SEEDS" GPUS="$GPUS" bash iterate/scripts/run_stage2.sh $TASK \
    $CODEGEN_SIM_AGENT_ROOT/evosearch/$TASK/evosearch_best_code.py \
    "$STAGE2_OUT"
```

It freezes the program (sha256), checks the stage markers (a contract violation aborts before any seed is
spent — fix the markers, not the seeds) and runs the assigned unused set once. Launch it with
`run_in_background=True` and poll; do not let your turn end while it runs. If it is killed mid-flight, finish
it with `RESUME=1` prepended; never start a second run into the same directory. Read the result in distinct
scenes: `$ROBOTWIN_PY iterate/scripts/stage2_report.py "$STAGE2_OUT"`.

**5c — write `$CODEGEN_SIM_AGENT_ROOT/evosearch/$TASK/findings.md`:**

```markdown
# <task> — evolutionary search findings

## Result
| | debug (10) | held-out clean (5) | held-out randomized (10) | distinct scenes |
|---|---|---|---|---|
| fix loop  | <n> | <n> | <n> | <n> |
| evosearch | <n> | <n> | <n> | <n> |

## What the fix loop was missing
<the structural difference, tied to check_success / play_once>

## Search
- iterations: <n> · stopping reason: <solved|max_iterations|blocked>
- winner: iter_NN / candidate_X
- hypotheses eliminated: <one line each, with the trace evidence>

## Generalizable patterns
### <short title>  (→ localize.md | grasp.md | transport.md | manipulation.md)
<the pattern, with the code snippet, phrased for any task — not this one>

## Collection suitability
1. Generalization: clean <n>/5 vs randomized <n>/10; fitted or not, and why
2. Program quality: does the code express the task's semantic structure
3. Stability: recoverable vs unrecoverable failures
4. Verdict: strong | marginal | unsuitable — one sentence of reasoning
```

The **Collection suitability** heading is required and must be last: the progress tooling keys on it to
tell a finished write-up from a truncated one.

---

## Return

```
TASK: <task>
Fix-loop held-out:  clean <n>/5, randomized <n>/10
Evosearch held-out: clean <n>/5, randomized <n>/10 (<n> distinct scenes)
Stopping reason: <solved|max_iterations|blocked>   Iterations: <n>
Winner: iter_NN/candidate_X — <one line: the strategy>
Skill candidates: <titles only>
Collection verdict: <strong|marginal|unsuitable>
```
```
