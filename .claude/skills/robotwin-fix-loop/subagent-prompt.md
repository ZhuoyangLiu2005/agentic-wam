---
name: robotwin-fix-loop-subagent-prompt
description: Stage-1 subagent template for any RoboTwin task. Fill TASK / BASELINE / SUBAGENT_GPUS, pass to the Agent tool with subagent_type general-purpose, run_in_background true.
---

# RoboTwin fix loop — Stage-1 subagent template

Copy everything between the markers, fill the variables at the top, dispatch with
`subagent_type: "general-purpose"`, `run_in_background: True`.

<!-- ==================== TEMPLATE START ==================== -->

## Task assignment

TASK:           <robotwin task name>
BASELINE:       $CODEGEN_SIM_AGENT_ROOT/baseline/demo_clean/<TASK>/<model>/run
SUBAGENT_GPUS:  <space-separated GPU indices assigned by the coordinator>

Working directory: the `codegen/sim` root. Run `source env.sh` first and every simulator command under
`$ROBOTWIN_PY`. Your task directory is `$CODEGEN_SIM_AGENT_ROOT/fix_loop/$TASK/`.

## What you are

A Stage-1 fix-loop subagent for one RoboTwin task (dual-arm ALOHA-AgileX, SAPIEN, privileged API). You
diagnose the baseline failures, author one generalizable program, and report findings.

**Your program is a data-collection oracle, not just a task solver.** The frozen program's episodes become
training data for an action model, and the style reference for that data is RoboTwin's own expert demo,
`play_once`. A program that succeeds through retries, park-and-regrasp detours, routine homing or staging
moves teaches the downstream model a hesitant, wandering policy. Success rate and motion economy are both
deliverables.

**Seeds.** Iterate on all ten debug seeds: **101-105** (`demo_clean`) and **151-155** (`demo_randomized`).
Developing against both is the point: a program that only sees clean scenes silently fits their appearance
and geometry, and the data it later generates teaches those artifacts. A round means running your current
candidate on all ten.

The locked validation seeds (201-270) are **off limits**: running even one voids this task's benchmark. The
coordinator runs them once, after you return, and that held-out score ranks tasks for collection. Do not
chase a perfect debug score with scene-specific tricks: they do not survive Stage 2, and an honest 8/10 beats
a fitted 10/10.

**Clean-room rule.** The fix must be authored by you from your own diagnosis. Do not read, copy or adapt
existing solution programs: frozen programs under `frozen/`, promotion history under `iterate/local/promoted/`, other tasks' `fix_code.py`, evosearch
candidates, or anything in the RoboTwin checkout other than `envs/<task>.py`. You *may* replay an existing
program to measure it; say in `findings.md` that you replayed rather than read it. Reading the shared skill
library (`../../.claude/skills/robotwin-program-authoring/`) is allowed and expected.

**Forbidden in program code** (invalidates the result): any access to `env.task`, SAPIEN actor handles,
`config["scale"]` or extents, contact points, `check_success` internals, or RoboTwin asset and config files
for geometry. The privileged API already mediates every legitimate ground-truth read.

**Read the task definition before you debug.** `capsim/third_party/RoboTwin/envs/<task>.py` is required
reading: the prohibition above is on what the *program* may call at run time, not on what *you* may read
while diagnosing. Two functions settle most questions:

- **`check_success()`**: the exact success condition. Enumerate every conjunct before you write a line of
  code. It routinely constrains objects the task name never mentions; several tasks require the *container*
  to be held off the table, not merely the payload placed into it.
- **`play_once()`**: RoboTwin's own expert demo, a worked solution. If your program is structurally
  different from `play_once`, that difference is usually the whole bug.

When a run scores 0 while everything you can observe looks correct, "the checker must be broken" is the
natural conclusion and almost always the wrong one. Open the source first.

**The line you must not cross:** use the source to understand *what the task requires* (which objects end
where, what must still be held at the end, which arm can reach). Never fit to the checker itself: no
`eps`/threshold literal copied out of `check_success`, no branch on values that only exist there, no
targeting a tolerance boundary. Every pose and dimension your program uses must still come from perception.
A program that satisfies the condition because it understood the task transfers; one that satisfies the
arithmetic does not. The same line applies to `play_once`'s numbers: its *structure* (which arm, what order,
where things go conceptually) is your style reference; its literal target coordinates are ground truth you
cannot observe. Constants pass this test only if they are properties of the robot (approach standoffs, lift
heights, jaw-sweep clearance, both-arm reach bands). Anything that locates a scene object or a placement
target must be derived from perception or the task instruction at run time.

## Motion economy — align your program's structure with `play_once`

After your candidate reaches a stable success rate (not before; correctness first), audit one round against
the expert demo's structure and close the gap:

- **Count gripper open/close cycles per arm** in your episodes and in `play_once`. Every extra cycle is a
  redundancy with a name (a handover the expert does not do, a park-and-regrasp, a staging move) and usually
  traces to one geometric decision, such as a placement line the far arm cannot reach or targets that
  collide with unmoved objects. Fix the decision, not the symptom. Where `play_once` itself uses the extra
  cycle (true handover tasks), yours should too.
- **Homing.** The expert returns an arm home only when the other arm takes over. Per-object homing and
  end-of-episode homing are pure frames.
- **Placement targets.** Choose them from perception so that no slot ever conflicts with an unmoved object;
  a geometry that needs a pre-clearing pass is a worse geometry, not a longer program.
- **Free retries.** A failed `goto_pose` plans without moving, so candidate ladders (alternate yaws, approach
  heights, re-sampled grasps) cost no frames. Prefer them over executed recovery motions. Verify before
  irreversible acts: a release can be checked by re-reading the object pose while still holding.
- **Count actions per arm, and know the floor.** `play_once` drives both arms with one concurrent move;
  `goto_pose(hand, ...)` is single-arm, so an identical action ledger still costs more frames than the
  expert. That residue is an API-expressiveness floor, not a redundancy. Match the ledger; note the
  concurrency gap in `findings.md` and do not burn rounds on it.
- **Gripper actuations cost frames, and some are no-ops.** Jaws are open at reset and after every release,
  so a defensive pre-grasp `open_gripper` is pure frames. Gate on the observed jaw state (the gripper entry
  of each arm's block in `get_observation()["robot_joint_pos"]`) instead of opening blind. Cycle counts
  alone will not reveal this; count actuations.
- **Resolving an unobservable discrete choice.** When the task hinges on a discrete fact you may not read
  (which half of a container each payload belongs in, which of two symmetric targets is "the" target), a
  reward-only A/B sweep over the debug seeds is the sanctioned way to settle it: run both variants, keep the
  one that scores, and disclose the sweep in `findings.md`. That is inference from the reward signal, not
  from privileged state.
- **Report the numbers.** Your final round's frames/episode and cycle counts go into `findings.md`; the
  coordinator ranks tasks for collection on them.

## Trainability — a program can pass the success gate and still be unusable

Success rate is a necessary check, not a sufficient one: a program can clear Stage 2 and still produce
episodes a policy cannot learn from. Before you freeze, audit the dimensions below and write what you found,
with numbers, into `findings.md`.

**Which dimensions bite depends on the task's success criterion.** Read `check_success()` and classify it
first: a *tolerant* criterion (a threshold with centimetres of slack — rows, "on top of", "inside the bin")
is forgiving; a *contact-critical* one (nesting, press-fit, balance, "seated") is where trainability dies.
Say in `findings.md` which class your task is, then audit accordingly.

- **Outcome stability under repetition** (matters most for contact-critical criteria). Does the same
  motion reliably produce the same outcome? Re-read the placed object after the arm retreats and compare
  with its pose at release. Post-place drift should be near zero; a median above about a centimetre, or a
  noticeable fraction of episodes drifting several centimetres, means the outcome is a coin flip, so two
  visually identical episodes carry opposite labels and the policy gets contradictory supervision. For a
  contact-critical task, aim the placement to land with visible margin rather than at the tolerance
  boundary, and verify (re-read while still holding) before the release. Measure drift on the object the
  placement could *disturb* (for a stack or an insertion, the base), and say which one you measured.
- **Decisiveness: one grasp per object.** Repeated grasps of the same object mean the demo contains
  failure-and-recovery, and the policy learns to expect the struggle. Prefer free planning ladders
  (§ Motion economy) over executed recovery. Report grasps-per-object.
- **Grasp style: jaws on faces, not edges.** Where the object has a definable orientation, align the wrist
  to it so the jaws close on opposite faces. A fixed top-down yaw edge-pinches a rotated cube: it hangs
  tilted and lands askew, and even when the task still passes the demo no longer matches how the expert, or
  your other tasks' data, does it. If the object has no meaningful yaw (spheres, round bowls), say so.
- **Stage markers: the program must slice.** The frozen program's text is what the code expert learns to
  write, one stage at a time. Mark the main body with `# [setup]` / `# [stage k/N] <title>` /
  `# [stage each] <title>` / `# [omit]` as specified in [../program-contract.md](../robotwin-program-authoring/program-contract.md):
  sequential top-level phases, retry ladders inside their stage, object loops as `for` over an order
  computed in the setup, a return description on every helper stub. `run_stage2.sh` refuses to freeze a
  program that fails `python3 $SLICER check`; run the check yourself before you return.

## Known pitfalls (do not spend a round rediscovering them)

- **Getting `play_once`'s frame count.** Boot `RoboTwinEnv` (`capsim/envs/simulators/robotwin.py`) and call
  `env.task.play_once()` under the same frame hook that records your own episodes (`_install_frame_hook`
  captures both at one cadence). Without the number, "align with the expert" is unfalsifiable.
- **Report distinct scenes, not just seeds.** Unstable scenes are retried as `seed+1` under the requested
  seed's name, task-specifically. Diff your per-seed perception printout across the set; identical object
  poses to four decimals means one scene. State the distinct count in `findings.md` next to the raw score.
- **Before you conclude a task is impossible, check the checker.** Some RoboTwin tasks are written around
  their own `play_once`: it assigns state to `self` mid-execution that `check_success()` then reads. On the
  agent path `capsim/integrations/robotwin/task_shims.py` supplies that state for the known tasks, and
  `task_completed()` prints `check_success() raised ...` to stderr (it lands in the trial log) whenever a
  checker still fails. When you see a flat 0/10 with no geometric failure, look for that line first, then
  call `env.task.check_success()` directly in a probe and read the exception. A defect here is an
  ENVIRONMENT class, not a strategy or capability limit; report it as such with the missing attribute, so a
  human can add the task to the shim.
- **Never derive an in-hand offset from one fixed camera.** Re-reading a held object from a single
  down-forward camera is biased along the camera ray, so the "correction" is noise wearing a plausible sign.
  If you grasped at the object's measured centre, the fingertip pose already is the object-centre pose.

## The API

Bare calls, already in the sandbox namespace (full contract: the docstrings in
`capsim/integrations/surfaces/arm.py` and `gripper.py`, which are the prompt every tier renders):
`get_observation()` · `get_object_pose(name)` → `(pos, quat_wxyz)` or `(None, None)` ·
`sample_grasp_pose(name, hand=None)` (raises when unreachable) ·
`get_object_3d_points_and_masks_from_language(text, use_multiview=True)` ·
`get_oriented_bounding_box_from_3d_points(points)` · `get_top_down_grasp_from_obb(obb, hand=None)` ·
`goto_pose(hand, position, quaternion_wxyz, z_approach=0.0)` → bool · `open_gripper(hand)` /
`close_gripper(hand)` / `goto_home_joint_position(hand)` · numpy (import it explicitly), scipy.

Conventions: `hand` is `"left"` / `"right"`; world-frame metres and WXYZ quaternions; motion functions
return `False` when planning fails; the two arms do not avoid each other, so home the idle arm before moving
the other one across the table. The robot-control rules the library has accumulated (near-arm rule, lift
before horizontal travel, `z_approach` on descents, never release mid-air on failure) are in
`../../.claude/skills/robotwin-program-authoring/transport.md`.

## Step 0 — Read the shared skills first

`../../.claude/skills/robotwin-program-authoring/{localize,grasp,transport,manipulation}.md` carry patterns other tasks already paid for. Use
them; you are not starting from zero. You may not edit them; your channel is `findings.md`.

## Stage 1

1. **Diagnose offline** (no GPU): for each baseline seed read `code.py`, `trace.json` (results nest as
   `{"result": {"result": ...}}`), `summary.txt` (program stdout: perceived poses etc.), and extract several
   frames from `video_combined.mp4` (25/50/75/100 %) with imageio and view them with the Read tool. A
   last-frame-only look misattributes mid-trajectory failures. Establish per seed what end state the task
   needs versus what the program left, checked against `check_success()`.
2. **Author candidate v1.** Seed-generic only: no per-seed branches, no fitted absolute coordinates; derive
   targets from perceived poses and extents.
3. **Round evaluation, all ten seeds in parallel** (one process per GPU you own, cycling if you have fewer
   GPUs than seeds). Note the two configs: 101-105 use the clean base, 151-155 the randomized one:

   ```bash
   # clean seeds
   CUDA_VISIBLE_DEVICES=<gpu> $ROBOTWIN_PY iterate/scripts/replay_trial.py \
     --code <candidate.py> --seed <101..105> --task $TASK \
     --config iterate/configs/priv_clean.yaml \
     --output-dir $CODEGEN_SIM_AGENT_ROOT/fix_loop/$TASK/attempts/round<K> &
   # randomized seeds
   CUDA_VISIBLE_DEVICES=<gpu> $ROBOTWIN_PY iterate/scripts/replay_trial.py \
     --code <candidate.py> --seed <151..155> --task $TASK \
     --config iterate/configs/priv_randomized.yaml \
     --output-dir $CODEGEN_SIM_AGENT_ROOT/fix_loop/$TASK/attempts/round<K> &
   ```

   The reward is in the produced trial directory name (`..._reward_1.000` = success); results land under
   `<output-dir>/oracle/…`. Use a distinct output dir per round; same-name trial dirs overwrite. Expect
   roughly 40 s of boot plus the task's runtime per replay; the runner kills a trial after 1000 s, so no
   unbounded retry loops. Program stdout lands in the trial's `summary.txt`, so print perceived poses,
   computed targets and final object poses; that is the cheapest diagnosis channel.
4. **Iterate**: diagnose the round's failures, revise, next round. **Budget: 4 rounds max.** Then stop,
   whatever the score.
5. **Deliver**
   - `$CODEGEN_SIM_AGENT_ROOT/fix_loop/$TASK/fix_code.py`: the best program (write one even if it scores 0;
     the campaign needs a program per task).
   - `$CODEGEN_SIM_AGENT_ROOT/fix_loop/$TASK/findings.md`: per-seed root causes; what fixed what, with
     round evidence; `## Generalizable patterns` (each: **Trigger** / **Code**, 5-20 real lines with
     constants generalised / **Evidence**, which seeds flipped / target skill file); blocked seeds;
     `## Infrastructure notes`; `## Clean-room attestation`; and, last, `## Collection suitability`
     answering in this order:
     1. **generalization**: the clean-versus-randomized split of your debug score, and your honest read on
        whether anything in the program is fitted to the ten debug scenes;
     2. **program quality**: does the code express the task's semantic structure (so code↔trajectory
        alignment is meaningful), or does it solve it with a couple of opaque motions? any constant you
        could not justify from perception?
     3. **stability**: how many attempts typically fail for recoverable versus unrecoverable reasons;
     4. **motion economy**: final-round frames/episode (mean over succeeding seeds) and per-arm
        gripper-cycle counts versus `play_once`'s, plus any executed motions with no counterpart in the
        expert demo (homing, parking, re-grasps) and why they remain;
     5. **verdict**: strong | marginal | unsuitable, one line why.

   The `## Collection suitability` heading must be present and last: the ledger uses it to tell a finished
   write-up from a truncated one.

## Return exactly

```
TASK: <task>
Rounds used: <k>/4
Final fix_code reward on debug seeds: clean 101-105 <k>/5, randomized 151-155 <k>/5 (total <k>/10, <n> distinct scenes)
Per-seed: 101:<0|1> 102:… 151:… 155:<0|1>
Seeds blocked: <list + root cause, or none>
fix_code.py / findings.md written: yes/no
Motion economy: ~<frames>/episode; gripper cycles <ours> vs play_once <theirs>; redundant executed motions: <none | list>
Collection suitability: strong | marginal | unsuitable — <one line why>
Trainability audit: criterion class (tolerant|contact-critical); post-place drift; grasps-per-object; grasp style — with numbers
Failure class if below the bar: STRATEGY | CAPABILITY | ENVIRONMENT — <one line>
Key findings:
  <1-3 bullets>
```

<!-- ==================== TEMPLATE END ==================== -->
