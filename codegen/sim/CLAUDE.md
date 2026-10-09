# codegen/sim — RoboTwin frozen-program pipeline (constitution)

This file is the constitution for agents and humans working in `codegen/sim`. Read it fully before running
or editing anything. Suite-specific runbooks live under `.claude/` and are indexed in
[`../../.claude/README.md`](../../.claude/README.md).

**Never commit or push unless the user asks.** A human reviews and pushes.

---

## Scope and reference documents

This workspace generates and validates RoboTwin programs (Stage A), then collects code-aligned
training data (Stage B). Training and policy evaluation belong to the external trainer repository.

Use the reference matching your task:

| Need | Reference |
|---|---|
| Project overview and quick start | [README.md](README.md) |
| Setup, execution commands, validation, promotion and recovery | [User guide](../../docs/ROBOTWIN.md) |
| Script and implementation locations | [Repository structure](../../docs/ROBOTWIN.md) |
| Agent roles and dispatch procedures | [Workflow](../../docs/ROBOTWIN.md), [.claude index](../../.claude/README.md) |
| Robot API signatures and privilege semantics | [API tiers](../../docs/ROBOTWIN.md), `capsim/integrations/surfaces/` |
| Training-program stage markers | [Program contract](../../.claude/skills/robotwin-program-authoring/program-contract.md) |
| Frozen-program and dataset fields | [Handoff contract](../../docs/ROBOTWIN.md) |

Read the relevant runbook before executing a stage. This file owns project-wide agent constraints;
reference documents own commands, file maps and data schemas.

## Environment and runtime constraints

Run `source env.sh` and `codegen_sim_preflight` before executing jobs. Configure machine paths in
ignored `site.yaml`; [site.example.yaml](site.example.yaml) defines the settings. Existing environment
variables take precedence. Use `ROBOTWIN_PY` for simulator commands. Follow the
[user guide setup](../../docs/ROBOTWIN.md) for interpreter and slicer
requirements; preflight alone does not verify assets, services or downstream dependencies.

### Security boundary

Programs execute with `exec` inside the trial process with full import access
(`capsim/envs/tasks/base.py`). Trial timeouts are not a security sandbox. Treat generated code as untrusted:
run workers on hosts without credentials or sensitive mounts, and keep API keys out of worker arguments.

### Runtime rules

- **Never set `CUDA_VISIBLE_DEVICES` inside a process** that renders through EGL/Vulkan; frames come back as
  garbage. Set it on the command line of the process (the scripts do).
- The sapien warning "Failed to find Vulkan ICD file" is benign; sapien ships its own ICD.
- Recording rollouts write a per-frame scratch cache. It is redirected off network storage automatically
  (`CAPX_ROLLOUT_CACHE` overrides the root). The cache is host-local, so run one collection `save_dir` from
  one host only.
- **hdf5 camera JPEGs are cv2-BGR-encoded.** Any non-cv2 decoder must swap R and B; `capsim/data/hdf5.py` is
  the one place that does it. Live `get_observation()` images are true RGB.
- One replay or collection worker uses about 6 GB of GPU memory. The scripts treat a card as usable at
  12 GB free (`MIN_FREE_MIB` in `run_stage2.sh`).
- **Some RoboTwin checkers read state only `play_once` sets** (`self.arm_tag`, `self.origin_z`). On the
  code-as-policy path that raised inside `check_success()` and scored every program 0.
  `capsim/integrations/robotwin/task_shims.py` supplies those attributes from the scene after each reset
  (the grasping arm, the object's height at reset); `task_completed()` now prints any checker exception
  to stderr instead of hiding it. A task that still prints `check_success() raised` needs an entry there.
- Actors a task keeps in a list (`self.bottles`) are exposed to privileged grounding as `<attr><i>`
  (`bottles1`, `bottles2`, ...), so "bottle" / "the largest bottle" resolve over them. Programs should
  NOT use those simulator names: they mean nothing to the perception tier. When a scene holds several
  instances of one asset, ask for them by position — "the leftmost / middle / rightmost <noun>" — which
  both tiers resolve geometrically (privileged ranks the matched actors by live world x, perception ranks
  the enumerated instances the same way), counting only what is inside the workspace. A spatial ordinal
  names a bowl while it RESTS; a bowl in the jaws or in flight must be tracked from the hand instead,
  because carrying it across the others re-ranks it. `capsim/integrations/robotwin/object_names.yaml`
  maps natural-language descriptions onto ground-truth actors inside the privileged tier.

## API usage constraints

Programs must use the registered robot surface documented in [API tiers](../../docs/ROBOTWIN.md).
Privilege changes the information source, not the failure contract: handle `(None, None)` from
`get_object_pose`, exceptions from `sample_grasp_pose`, and failed motion planning. Gripper-close
completion does not prove a secure grasp. The two arms do not avoid each other.

Use world-frame metres and WXYZ quaternions for motion poses; `hand` is `"left"` or `"right"`.
Import numpy explicitly when needed. Consult the declarative surface docstrings for full signatures.

## Forbidden in program code

A program may not reach into `env.task`, SAPIEN actor handles, `config["scale"]` or extents, contact points,
`check_success` internals, or RoboTwin asset and config files. Those are privileged-API territory. The test is
simple: if a real robot with a camera could not do it, the program must not do it. Review programs against
this boundary before validation; local diagnostic scripts are not dependencies of the public workflow.

**Run time versus debug time.** The rule governs what a *program* may call while it executes. An agent may
read `capsim/third_party/RoboTwin/envs/<task>.py` while diagnosing: `check_success()` is the only statement of
what the task demands and `play_once()` is a worked solution. Use the source to learn *what the task
requires*; never fit to the checker. Concretely: no threshold literal copied out of `check_success`, no branch
on a value that exists only there, no aiming at a tolerance boundary. Every pose and dimension a program
uses still comes from perception or the task instruction at run time. Constants are legitimate only when they
are properties of the robot (approach standoffs, lift heights, jaw clearance, reach bands).

## Program contract (training data)

A program meant for code-stream training carries stage markers — `# [setup]`, `# [stage k/N] <title>`,
`# [stage each] <title>`, `# [omit]` — and sized stage bodies, as specified in
[`../../.claude/skills/robotwin-program-authoring/program-contract.md`](../../.claude/skills/robotwin-program-authoring/program-contract.md). `run_stage2.sh` refuses to freeze a marked
program that fails `$SLICER check`; `collect_aligned.sh` builds the slicing cache when a collection ends.

## Seed policy

| Purpose | Seeds | Who runs them |
|---|---|---|
| Debug (iterate freely) | 101-105 `demo_clean`, 151-155 `demo_randomized` | every fix-loop or evosearch round |
| Locked validation (one shot per program) | default 201-205 clean + 251-260 randomized | `iterate/scripts/run_stage2.sh` only |
| Collection | 1001+ clean, 2001+ randomized | `pipeline/scripts/collect_aligned.sh` / `make_dataset.py` |

- Never run a locked seed while iterating. `evosearch_eval.py` refuses seeds 201-210 and 251-270;
  `collect_aligned.sh` refuses any start seed below 1001 (lift with `ALLOW_RESERVED=1` only for a post-freeze
  official-parity collection).
- Stage 2 is one shot per program *and* per seed range: `run_stage2.sh` refuses to run into an `out_root`
  that already has results (`RESUME=1` only finishes an interrupted run of the same sha256). A later program
  for a task whose range was spent mints a fresh range (`CLEAN_SEEDS="..." RAND_SEEDS="..."`) and a new
  `out_root`, chosen outside the debug and collection bands and never one already reported on. Only the fixed
  bands above are guarded by the scripts; the operator enforces the rest.
- A replay's seed range comes from `resume_idx` in the YAML plus `--total-trials N` (inclusive end).

> **Seeds are not scenes.** `RoboTwinEnv.reset` retries an `UnStableError` scene with `seed+1` while keying
> the episode by the *requested* seed, so consecutive seeds can land on one scene, and which ones collapse is
> task-specific. Read every score in distinct scenes (`iterate/scripts/stage2_report.py`)
> and report the scene estimate next to the seed count. The reporter uses rounded perception printouts,
> not complete simulator state; missing fingerprints and collisions limit what it can measure.

Before promotion, verify all scheduled trials completed, the success gate is met and the program passes
the trainability review. `freeze.py` records matching validation metadata but does not enforce PASS;
`make_dataset.py` does not enforce the handoff verdict or checksum. The default report PASS label assumes
15 trials and at least 12 successes. For a different trial count, review the rate and completeness explicitly.

## Collection constraints

Collect only from the reviewed, validated program. Do not edit frozen programs in place or mix
program versions or clean/randomized protocols in one output directory. The normal handoff is
`frozen/<task>/task.yaml`; a `--program` override requires the same validation review.

Follow the [collection checklist](../../.claude/skills/robotwin-collect/INSTRUCTIONS.md). Completion requires verified
kept-episode counts and usable code slices, not merely the existence of `dataset.yaml`.
The [handoff contract](../../docs/ROBOTWIN.md) defines artifacts and alignment fields.
