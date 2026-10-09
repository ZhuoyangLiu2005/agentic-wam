# RoboTwin (simulation)

Everything AgenticWAM does in the RoboTwin 2.0 simulator, in pipeline order: collecting code-labeled
demonstrations, turning them into training data, training, evaluating the WAM, and the **hybrid** evaluation in
which the code the model generates is executed on the robot whenever it passes the arbiter's gates, with the WAM
action head as the fallback — plus the viewer for those rollouts.

```
§2  codegen/sim     Stage A: write + validate a task program  -> frozen/<task>/
                    Stage B: execute it, record which line drove each motion -> LeRobot v2.1 + stage slices + dataset.yaml
§3  dataprep        dataset.yaml -> text-embed cache + configs/robotwin/{data,task}/<name>.yaml
§4  training        bash scripts/train_robotwin.sh <name>
§5  benchmark       robotwin/eval/benchmark.py: every task x easy/hard, sharded over GPUs -> report.py tables
§6  hybrid eval     frozen/<task>/ -> configs/robotwin/tasks/<task>/ ; code x WAM, arbitrated per plan step
§7  viewer          robotwin/hybrid_eval/rollout_viewer.py build | serve
```

**Where the code lives.**

```
codegen/sim/          program authoring + code-labeled collection (its own environment; §2)
robotwin/
├── dataprep/         collected data -> training configs: text-embed cache, data/task config writer, stage slicer
├── deploy/           the policy RoboTwin's eval calls (get_model / eval / reset_model), linked into
│                     RoboTwin/policy/<EVALUATION.policy_name>; the real-robot server reuses its policy class
├── eval/             benchmark.py (the whole benchmark), report.py (tables), run_task.py (one task in one process)
└── hybrid_eval/      launch.py (one eval arm, WAM or hybrid), the arbiter, task import, vision servers, rollout_viewer.py
configs/robotwin/     task/ (robotwin_wam, robotwin_hybrid) + data/ (robotwin_official, robotwin_cap): training configs
                      sim_robotwin.yaml (the Hydra root run_task.py composes: RoboTwin checkout, policy name, eval defaults)
                      site.yaml (this machine's paths; from site.example.yaml), benchmark.yaml (suites, protocols,
                      episode plans), models/ (eval presets), tasks/ + tasks_indep/ (hybrid task assets)
```

`scripts/posttrain_annotation_robotwin.sh <stage>` wraps §2, §3 and the task import of §6 (`baseline`, `replay`,
`evosearch`, `validate`, `freeze`, `plan`, `collect`, `prepare`, `import`); its header lists what each stage runs.

**Two training configs, one per data source** (`configs/robotwin/task/`). Both cover the 50-task suite, use 14-D
aloha-agilex joint actions (direct qpos, no IK) and the AgenticWAM model (`configs/model/agenticwam_cosmos.yaml`).

| task config | data config | data | code labels | evaluated with |
|---|---|---|---|---|
| `robotwin_wam` | `robotwin_official` | RoboTwin 2.0 official expert demonstrations (§3.3) | frozen caches from an earlier labeling route | the WAM benchmark (§5) |
| `robotwin_hybrid` | `robotwin_cap` | collected with `codegen/sim` (§2–3.1) | exact: the executed program, cut into stages | the WAM benchmark and the hybrid evaluation (§6) |

**Conventions used throughout**
```bash
W=<repo root>                                      # run everything from here
RT=<RoboTwin 2.0 checkout>                        # = paths.robotwin_root (configs/paths/default.yaml)
D=<RoboTwin 2.0 dataset dir>                      # = paths.robotwin_official: robotwin2.0/ + dataset_stats.json
export PYTHONPATH="$W/src:${PYTHONPATH:-}"         # MANDATORY — else `import agenticwam` resolves to another checkout
```

Related: [INSTALL.md](INSTALL.md) (environment), [CODE_ANNOTATION.md](CODE_ANNOTATION.md) (code labels for
real-robot data, the flat cache, and how the loader picks a sample's code).

---

## 1. Setup

There are three pieces, each with its own environment:

| piece | environment | setup |
|---|---|---|
| `codegen/sim` (collection, §2) | `codegen-sim` (+ `codegen-data`, optional `codegen-perception`) | `codegen/sim/SETUP.md`; RoboTwin checkout in `codegen/sim/capsim/third_party/RoboTwin` |
| training (§4) | the AgenticWAM env (INSTALL.md) | — |
| benchmark, hybrid eval, viewer (§5–7) | the AgenticWAM env | `configs/robotwin/site.yaml` from `site.example.yaml` (python env, RoboTwin checkout, Vulkan/FFmpeg libs, HF caches, vision ports, `viewer_out`) |

### 1.1 The RoboTwin checkout used for evaluation

Evaluation points at a RoboTwin 2.0 checkout
(config `EVALUATION.robotwin_root` in `configs/robotwin/sim_robotwin.yaml`):

```
$RT        # SAPIEN sim + aloha-agilex cuRobo assets
```

Installation lives in that upstream checkout, **not** in this repo:
```bash
$RT/script/_install.sh          # SAPIEN, cuRobo, python deps
$RT/script/_download_assets.sh  # scene/object/robot assets
$RT/README.md                   # official RoboTwin 2.0 install guide
```

- The model-side environment is the AgenticWAM env ([INSTALL.md](INSTALL.md): Python 3.10, CUDA 12.8, `transformers>=4.57`).
- Sim runtime needs **Vulkan + cuRobo**: `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`,
  `LD_LIBRARY_PATH=<vulkan lib dir>:…`, plus `DIFFSYNTH_*`, `HF_HOME`. `hybrid_eval/launch.py` sets them from
  `site.yaml`; for the direct entry point export them yourself (§5.4).
- The policy package `robotwin/deploy/` is **linked** into `RoboTwin/policy/<EVALUATION.policy_name>` (default
  `agenticwam_policy`, `configs/robotwin/sim_robotwin.yaml`) by `robotwin/eval/run_task.py` on first use. A link of that name that
  points at another checkout is refused; give this checkout its own `EVALUATION.policy_name` in that case.

For a fresh node, either follow the upstream RoboTwin 2.0 project (`_install.sh` + `_download_assets.sh` + its README)
or reuse the checkout `codegen/sim/SETUP.md` fetches, and point `EVALUATION.robotwin_root` (WAM eval) and
`robotwin_root` in `site.yaml` (hybrid eval) at it.

### 1.2 codegen/sim

Follow `codegen/sim/SETUP.md`: it creates the `codegen-sim` environment (simulation), `codegen-data`
(export) and optionally `codegen-perception` (Molmo / SAM3 / Contact-GraspNet), fetches the RoboTwin
checkout into `codegen/sim/capsim/third_party/`, and writes `site.yaml` from `site.example.yaml`. Then, in
every session, from `codegen/sim`:

```bash
conda activate codegen-sim
source env.sh
codegen_sim_preflight
```

The trainer the exporter and slicer need is this repository; with `TRAINER_ROOT` unset it is found
automatically. Artifacts default to `codegen/sim/outputs/{agentic,validation,data,logs}` (configurable in
`site.yaml`); `frozen/` and `outputs/` are git-ignored. Agent rules — what a program may read, and the
seed policy below — are in `codegen/sim/CLAUDE.md`.

---

## 2. Data collection — code-labeled demonstrations (`codegen/sim`)

### 2.1 How it works

In simulation the program *is* the label. A validated program is executed in RoboTwin, and the engine
records which program line issued every motion; training slices are cut from that record, so code and
actions agree by construction.

```
Stage A   LLM baseline → Claude Code fix loop → (evolutionary search) → held-out validation → freeze
          → frozen/<task>/
Stage B   frozen/<task>/ → aligned collection → LeRobot export + stage slices → dataset.yaml
```

`codegen/sim/capsim/` is the shared engine: simulator adapter, robot API, call tracing, model client and
data conversion. A model is used only to *write* programs in Stage A; baseline replays, validation and
collection execute programs without model calls.

### 2.2 Seeds

| band | clean | randomized | used for |
|---|---|---|---|
| debug | 101–105 | 151–155 | every Stage A iteration: baseline, fix loop, evolutionary search |
| held-out | 201–205 | 251–260 | one validation of the final candidate |
| collection | from 1001 | from 2001 | Stage B |

Never pick between candidates using held-out results. A revised program is validated on a fresh, unused
range with a new output root; record which ranges have been spent. `make_dataset.py` refuses collection
starts below 1001 unless `ALLOW_RESERVED=1`.

### 2.3 The robot API and privilege tiers

Programs target the dual-arm ALOHA-AgileX through one API. The gripper surface has 11 functions, and
their order is part of the prompt (declared in `capsim/integrations/surfaces/`; do not reorder):

`get_observation` · `get_object_pose` · `sample_grasp_pose` · `goto_pose` · `open_gripper` ·
`close_gripper` · `get_oriented_bounding_box_from_3d_points` · `get_top_down_grasp_from_obb` ·
`get_object_3d_points_and_masks_from_language` · `goto_home_joint_position` · `get_gripper_aperture`

`hand` is `"left"` / `"right"`; positions are world-frame metres, quaternions world-frame WXYZ.

How much ground truth the API uses is a dial:

| tier | API class (+ `Traced` variant) | object pose / segmentation | grasp | servers |
|---|---|---|---|---|
| 0 · perception | `RoboTwinGripperApi` | Molmo + SAM3 | Contact-GraspNet | Molmo :8122, SAM3 :8114, GraspNet :8115 |
| 1 · perception-privileged | `RoboTwinGripperPrivilegedPerceptionApi` | simulator ground truth | Contact-GraspNet on the clean segment | GraspNet |
| 2 · fully privileged | `RoboTwinGripperPrivilegedApi` | simulator ground truth | annotated grasp, reachability still checked | none |

Three invariants keep tiers comparable: every tier renders the same API docs; privilege lowers the
failure *rate* but never the failure *contract* (`get_object_pose` still returns `(None, None)` for an
unresolvable description, `sample_grasp_pose` still raises when no grasp is reachable); motion planning,
geometry and the gripper layer are the same real implementation everywhere. Programs name objects in
natural language and never read simulator actors, asset metadata or checker state. `Traced` variants log
every call to `trace.json`.

### 2.4 Stage A — write and validate a program

Paths are relative to `codegen/sim` (or use the matching stage of `scripts/posttrain_annotation_robotwin.sh`).

**Baseline** — a model writes programs with execution feedback (both debug settings; `--args.total-trials`
is the inclusive end seed):

```bash
"$ROBOTWIN_PY" iterate/scripts/run_robotwin_batch.py --args.base-config-path iterate/configs/priv_clean.yaml \
  --args.tasks <task> --args.resume-idx 101 --args.total-trials 105 \
  --args.server-url <chat-completions-url> --args.models <model>
"$ROBOTWIN_PY" iterate/scripts/run_robotwin_batch.py --args.base-config-path iterate/configs/priv_randomized.yaml \
  --args.tasks <task> --args.resume-idx 151 --args.total-trials 155 \
  --args.server-url <chat-completions-url> --args.models <model>
```

**Fix loop and evolutionary search** — run in Claude Code with the repository skills
`robotwin-fix-loop` and `robotwin-evosearch` (`.claude/skills/`). A coordinator gives each task to a
subagent, which studies the baseline failures, writes `fix_code.py` and `findings.md`, and tests on the
debug seeds; evolutionary search takes over for tasks stuck on a strategy-level problem. Both draw on
`robotwin-program-authoring`, the shared robot-control knowledge and program contract; only the
coordinator edits it. One debug replay, and a candidate population:

```bash
CUDA_VISIBLE_DEVICES=0 "$ROBOTWIN_PY" iterate/scripts/replay_trial.py --code <program.py> --task <task> \
  --seed 101 --config iterate/configs/priv_clean.yaml --output-dir <out>
"$ROBOTWIN_PY" iterate/scripts/evosearch_eval.py --iter-dir <dir with candidate_*.py> --task <task> --gpus 0 1
```

**Held-out validation and freeze** — once, on a new output directory:

```bash
GPUS="0 1" bash iterate/scripts/run_stage2.sh <task> <program.py> <validation-out>
python3 iterate/scripts/stage2_report.py <validation-out>
# write <validation-out>/STAGE2_RESULT.md with the score line:
#   clean X/A + randomized Y/B = Z/N — PASS|FAIL
python3 iterate/scripts/freeze.py <task> --program <validation-out>/frozen_program.py \
    --stage2 <validation-out>/STAGE2_RESULT.md
```

PASS needs every scheduled trial complete, at least 12 of the 15 successful, and the trainability audit
from the fix-loop skill. `run_stage2.sh` checks the program contract (§2.5) and aborts on a violation.
Neither `freeze.py` nor `make_dataset.py` enforces the verdict — reviewing it is the operator's job.
`RESUME=1` only finishes an interrupted validation of the same frozen program.

### 2.5 The program contract (stage markers)

A frozen program is also what the code expert learns to write, one stage at a time, so its main body
carries stage markers:

```
# [setup]              perception / planning before any motion
# [stage k/N] <title>  one meaningful unit of the task
# [stage each] <title> the next statement is a for-loop over objects; each iteration is a stage
# [omit]               audits and fallbacks that must not appear in training text
```

The rules that matter most: each stage body is **40–128 tokens and 3–12 lines** (it must fit the decode
canvas); flat, with at most one level of control flow; the intent goes in the stage title, because body
comments are dropped; one helper per motion phase, called with plain names, lookups and UPPERCASE
constants; wide results come back as dict records; guards test a bound result, never the motion call
itself; per-object work is a `[stage each]` loop over a precomputed order; nothing load-bearing goes
under `[omit]`. Objects that change per episode are written as the task's placeholders (`"{A}"`).
The full contract is `.claude/skills/robotwin-program-authoring/program-contract.md`; the slicer that
enforces it is §3.2.

### 2.6 Stage B — collect and export

```bash
python3 pipeline/scripts/plan_collection.py --kept <target> --tasks <task>          # how many seeds to attempt
python3 pipeline/scripts/make_dataset.py --task <task> --episodes <attempts> --gpus 0 1 \
    --task-config demo_clean --start-seed 1001
python3 pipeline/scripts/make_dataset.py --task <task> --episodes <attempts> --gpus 0 1 \
    --task-config demo_randomized --start-seed 2001 --out <randomized-out>      # separate output dir
```

`--episodes` counts attempted seeds; only successful episodes are exported. `--dry-run` prints the
resolved program, seeds and config. `--instructions auto` picks how per-episode instructions are
filled (static rule or expert rollout); `--n-instructions` caps exported paraphrases (default 10).
Run one output directory from one host (the saver cache is host-local) and give concurrent collections
separate `CODEGEN_SIM_LOG_ROOT`s. Afterwards check `dataset.yaml`: `episodes.kept`, `stage_markers`,
`records_have_top_line` and `slices` — `slices: null` means the slicing cache failed to build and the
dataset is not ready for code-stream training.

### 2.7 What Stage A and Stage B hand over

**`frozen/<task>/`** (written by `freeze.py`): `frozen_program.py`, `frozen_program.sha256`,
`STAGE2_RESULT.md` and `task.yaml` — `task`, `program`, `sha256`, `stages`, `template_slots`,
`candidates`, `record_instructions` (`static` / `expert` / `off`), `embodiment`, `stage2` (the matching
validation scores and verdict, or null).

**Collection output** (`$CODEGEN_SIM_DATA_ROOT/<task>/`):

```
collect.yaml
raw/data/episode<N>.hdf5   raw/video/   raw/manifest.jsonl
raw/alignment/episode<N>.json      per-motion provenance: line, top_line, top_locals, api, hand,
                                   frame_start, frame_end; plus the executed (materialized) code
raw/slices/                        preamble.txt  slices_ep/  cycle_by_idx.pt
                                   episodes_{train,val,all}.json  stage_frames.json
lerobot/robotwin2.0/               LeRobot v2.1 export; meta/episode_seed_map.json maps episodes to seeds
dataset.yaml
```

`dataset.yaml` fields: `task`, `task_config`, `created`, `program`, `program_sha256`, `stage_markers`,
`seeds.{start,end}`, `episodes.run` (raw manifest rows), `episodes.kept` (exported successes),
`raw_root`, `lerobot_dir`, `instructions`, `instructions_per_episode`, `records_have_top_line`, `slices`.
It is an index, not an audit — check files and alignment coverage before training.

Saved RoboTwin JPEGs are BGR. Decode raw episodes through `capsim/data/hdf5.py`; the exported videos are
already RGB, and live `get_observation()` images are RGB too.

### 2.8 When collection fails

* Missing imports or simulator assets — check `site.yaml`, the RoboTwin checkout and the Python in use.
* A task keeps failing — read the replay logs, traces and videos. Task source may be read to understand
  the goal; programs may only use the public API.
* `check_success()` raises — inspect the task adapter and `capsim/integrations/robotwin/task_shims.py`;
  a zero score alone does not tell a program error from a simulator defect.
* Validation scenes look repeated — report the report's scene estimate and missing fingerprints with the
  trial counts.
* Wrong colours in exported data — decode through `capsim/data/hdf5.py` (raw JPEGs are BGR).

---

## 3. Training data

### 3.1 From collected data to a training config (`robotwin/dataprep`)

The seam between collection and training is one manifest per collected task, `dataset.yaml` (§2.7), written by
`codegen/sim/pipeline/scripts/make_dataset.py`:

    task, program (+ sha256, stage_markers), seeds, episodes{run,kept}, raw_root, lerobot_dir,
    slices{preamble, slices_ep, cycle_by_idx, episodes_{train,val,all}, stage_frames} | null

`lerobot_dir` is a LeRobot v2.1 dataset (14-D joint state/action, three cameras 480x640, per-frame instruction =
RoboTwin's filled paraphrases); `slices` is the stage-slicing cache the code stream trains on, built at collection
time by `robotwin/dataprep/slicing.py` from the program's stage markers (§3.2).

Everything on this side lives in one package, `robotwin/dataprep/`:

| file | role |
|---|---|
| `slicing.py` | stage contract -> `slices_ep/ep_%06d.json`, `cycle_by_idx.pt`, `preamble.txt`; the collector calls it via `$SLICER` |
| `text_cache.py` | Cosmos-Reason1 prompt-embedding cache (required by the backbone; ~25 MB per distinct instruction) |
| `data_config.py` + `data_base.yaml` | one `RobotVideoDataset` block per manifest -> `configs/robotwin/data/<name>.yaml` |
| `task_template.yaml` | the finetune recipe; `prepare.py` fills `override /data`, foundation and ViT-prefix paths |
| `smoke.py` | instantiate a data config and fetch samples |
| `prepare.py` | the steps above in one command |

The trainer imports nothing from here; it only reads the files these scripts emit.

    python robotwin/dataprep/prepare.py --datasets <sim>/outputs/data/*/dataset.yaml --name my_tasks
    bash scripts/train_robotwin.sh my_tasks batch_size=4 max_steps=15000

`prepare.py` builds the text cache, writes `configs/robotwin/data/my_tasks.yaml` and `configs/robotwin/task/my_tasks.yaml`
(from `task_template.yaml`, the `robotwin_hybrid` recipe), and smokes
the data config. Machine paths (normalisation stats, text-cache disk, the Cosmos-Reason1 text encoder, SDAR, the foundation checkpoint, Qwen3-VL)
come from flags or the `dataprep:` block of `configs/robotwin/site.yaml` (template: `site.example.yaml`).
`--skip-text-cache` reuses an existing cache; `--split all` trains on every kept episode instead of the train split.

Text-cache size is set at export time: the LeRobot export cycles each episode's paraphrases over its frames
(`make_dataset.py --n-instructions`, default 10 = the first ten RoboTwin recorded), and every distinct paraphrase
costs one 25 MB file. Ten per episode gives ~500 distinct instructions per 60-episode task (~12 GB); one hundred
gives up to ~5,500 (~140 GB).

Verified 2026-09-06 end to end on click_alarmclock (3 episodes collected by `make_dataset.py` with the annotated
program): 202 windows, samples carry `prompt_embs` (128 x 100352), `policy_code` starting `# [stage 1/2] ...` and
`prompt_code` with the filled instruction.

### 3.2 How the code stream is cut — stage slicing

Engine: `robotwin/dataprep/slicing.py`; tests: `robotwin/dataprep/tests/test_slicing.py`
(synthetic programs, no data needed). The same slices are the training targets (§3.1) and the hybrid eval's
prompt and whitelist (§6).

#### The markers

The code stream (prelude + one slice per stage, with the frames each stage spans) is defined by comment
markers in the program's main body. There is no per-task slicer code.

    # [setup]                  following top-level statements form the prelude's `# Setup:` block
    # [stage k/N] <title>      following top-level statements form stage k's slice (`# [stage k/N] title` header)
    # [stage each] <title>     the next statement is a top-level `for`: every EXECUTED iteration becomes one stage
                               (one grasp-release cycle per object). The loop targets are bound to the iteration's
                               values in the slice text; the title is a template over the program's constants and
                               the loop variables, e.g. `# [stage each] pick and place {NAMES[idx]}`
    # [omit]                   following top-level statements are left out (dead fallbacks, audit prints)

Sections may be reopened; imports, UPPERCASE constants and `def`s are always prelude material.

**Prompt form (2026-09-10).** The preamble the code expert is prompted with (`preamble.txt`, embedded in every
`slices_ep/*.json`) is deliberately short: one `def name(args): ...` stub per helper that a stage body calls,
then one line `# Setup defines: a, b, c` naming the setup variables the stages read. Imports, constants, the
return descriptions and the executable setup are NOT in the prompt; the hybrid eval derives the executable
`[setup]` statements from the frozen program (`Program.exec_setup()`) and runs them after the skeleton.
`slicing.py preamble|build --style desc` keeps the `-> "desc"` return descriptions in the stubs, `--style full`
reproduces the legacy preamble (imports + constants + every stub + `# Setup:` block), `--all-helpers` lists every
helper, `--no-setup-names` drops the names line. The text after `->` in a helper's stub is a string return
annotation (`def f(...) -> "desc":`), else a real annotation kept verbatim, else the docstring's first line. `print()` calls (and calls to helpers whose body only prints) and `if ...: raise` guards are stripped (a guard whose test calls a helper or a motion API, `if not reach(...): raise`, is kept); module docstrings are dropped; an UPPERCASE assignment is a constant only when its value needs nothing from the main body. Text is
rendered with a pinned unparser (tuples always parenthesised), so caches are identical on every Python version.

`slicing.py check <task | program.py> [--executed frozen.py]` verifies: markers well-formed and contiguous,
each `[stage each]` followed by exactly one `for` over plain names, every slice (and every loop body) compiles,
no omitted statement issues a motion, `tasks/<task>/preamble.txt` is current, and — given the executed program
(`task.yaml: source.frozen_program`) — that the annotated file differs only in comments and annotations.

#### Frames: what the collector records

Each motion record in `alignment/episode<seed>.json` carries (aspire `code_alignment.py`, since 2026-09-06):

| field | meaning |
|---|---|
| `line` | innermost program line that issued the motion (inside a helper, usually) |
| `top_line` | line of the OUTERMOST program frame: the top-level statement that (through helpers) issued it |
| `top_locals` | the outermost frame's simple-typed locals (scalars, flat lists) — identifies loop iterations |
| `frame_start`, `frame_end` | video frames the motion spans |

A record belongs to the stage of the top-level statement at `top_line`; inside a `[stage each]` loop, to the
iteration whose loop-target values match `top_locals`. A stage instance spans from its first record to the next
instance's first record (the first starts at frame 0; an instance that never moved is zero-length where the
next begins). Alignment files without `top_line` (collections before 2026-09-06) fall back to a heuristic that
is exact unless consecutive stages share a helper.

#### Where it runs in the pipeline

- **Freeze** (`codegen/sim/iterate/scripts/freeze.py`): a program with `# [stage` markers is checked
  statically after being frozen; a violation aborts the freeze; the section listing is kept in `stages.txt`.
  Unannotated programs only get a warning. (`SLICER=` points at `slicing.py` if the repos are not side by side.)
- **Collection** (`codegen/sim/pipeline/scripts/collect_aligned.sh`): at the end, `slicing.py build` writes the
  code-stream cache next to the data (`<save_dir>/slices/`) — the collection ships with its own slicing catalog
  (`stage_frames.json`: per-episode stage frame ranges, empty instances reported).
- **Training**: a data config points at the cache: `preamble.txt`, `slices_ep/ep_%06d.json`, `cycle_by_idx.pt`,
  `episodes_{train,val,all}.json`. Header word is `stage`; `assemble.parse_header` also accepts `step`.
- **Deploy**: `tasks/<task>/preamble.txt` (the prompt, written by `slicing.py preamble <task> --write`) and the
  annotated `frozen_program.py` are the only task inputs of the hybrid eval; `assemble.setup_block()` derives the
  executable setup from the frozen program.
  `robotwin/hybrid_eval/import_tasks.py --frozen-root codegen/sim/frozen` fills
  `configs/robotwin/tasks/<task>/` from every frozen task (program copy, prompt
  preamble, `task.yaml` with sha256, per-task `gen_tokens` = longest slice in SDAR tokens + 8 rounded to 8,
  `nis` at the preset's 40/96 ratio, and the Stage-2 score when `--scores-root` is given).
- Helpers whose body only prints (`def log(*a): print(*a)`) are logging: their calls are stripped from the
  slices like `print()`, so they are not stubbed in the prompt either.

    python robotwin/dataprep/slicing.py check    <task>
    python robotwin/dataprep/slicing.py preamble <task> --write
    python robotwin/dataprep/slicing.py build    --task <task> --align-dir <dir> --out <cache dir>
    python robotwin/dataprep/slicing.py regress  --task <task> --align-dir <dir> --legacy <old cache>

#### Status (2026-09-06)

| task | markers | preamble = legacy | slices = legacy | frames = legacy |
|---|---|---|---|---|
| click_alarmclock | fixed stages | yes | yes | 60/60 |
| beat_block_hammer | fixed stages | yes | yes | 59/59 |
| place_a2b_left | fixed stages | yes | yes | needs `top_line` (shared helper `reach`) |
| blocks_ranking_size | `[stage each]` | yes | body keeps the `if h is None` guard the old slicer resolved by hand | needs `top_locals` |
| blocks_ranking_rgb | none | legacy | `while remaining:` with a data-dependent order: not a `for` — rewrite or keep legacy | legacy |

Legacy caches stay authoritative for checkpoints trained on them. Open: re-collect a2b/brs with the new
collector fields, rewrite brgb's loop as a `for` over a precomputed order, annotate the 34 Stage-2 PASS
programs (agent-drafted markers, `check --executed` as the gate), then collect.

### 3.3 The RoboTwin 2.0 official dataset (`robotwin_official`)

The RoboTwin 2.0 expert demonstrations, pre-built in LeRobot format (14-D joint), with the code labels and prompts of
the RoboTwin runs. `configs/robotwin/data/robotwin_official.yaml`:

| | path |
|---|---|
| LeRobot dataset | `${paths.robotwin_official}/robotwin2.0/` |
| normalisation stats | `${paths.robotwin_official}/dataset_stats.json` |
| code labels, prompts, text embeddings | `${paths.robotwin_official_labels}/`, laid out as below |

```
${paths.robotwin_official_labels}/
├── train/                          50 tasks x 50 clean episodes
│   ├── episodes.json               the episode selection
│   ├── episode_instruction.json    per-episode instruction
│   ├── skill_prompt_by_instruction.json
│   ├── policy_code_idx.pt          code labels (flat cache)
│   └── text_embeds/                Cosmos-Reason1 instruction embeddings
├── val/                            the same five, for the validation episodes
├── policy_code_cache/              per-instruction code (legacy lookup)
├── policy_code_slices/             per-object code slices
├── cycle_by_idx.pt                 slice-cycle index of every frame
├── task_instruction.json           canonical instruction per task (evaluation)
└── task_skill_prompt.json          skill prompt per task (evaluation)
```

#### Rebuilding the text-embed cache
```bash
# Cosmos-Reason1 text-embed cache (once, on a GPU node)
python scripts/precompute_cosmos_text_embeds.py \
  --instructions <instructions.txt> \
  --out-dir      <text_embeds dir> \
  [--ckpt <Cosmos-Reason1-7B dir>]   # default: paths.text_encoder \
  --max-len 128           # optional: --limit N, --device cuda
```

These code labels come from an earlier RoboTwin labeling route (removed with its cache builders; git history). They
are frozen artifacts and cannot be rebuilt from this repository; new RoboTwin labels come from `codegen/sim` (§2).

---

## 4. Training

Entry point: `bash scripts/train_robotwin.sh <task> [hydra overrides]` — accelerate + DeepSpeed ZeRO-1 on
`NPROC` GPUs (default 8), weights-only checkpoints. It runs `scripts/preflight_sft.py` first (the config
composes, every path exists, the code cache covers the selected episodes, GPUs are idle), then
`scripts/launch_train.py`. Output goes to `./runs/<task>/<RUN_ID>` (`RUN_ID` defaults to a timestamp; set it
to resume into a fixed directory) and wandb is named `<task>`.

```bash
export HF_HOME=/path/to/hf_cache
export PYTHON=/path/to/conda/envs/<agenticwam-env>/bin/python      # else `python` on PATH
# WANDB_API_KEY comes from the environment ONLY — never write it into a script/config/log.

bash scripts/train_robotwin.sh robotwin_wam                      # official data: 200k steps, batch 2 x 8 GPUs
NNODES=3 NODE_RANK=<r> MASTER_ADDR=<m> MASTER_PORT=<p> \
  bash scripts/train_robotwin.sh robotwin_hybrid                 # codegen/sim data: 15k steps, 3 nodes x 8 x batch 4
```

`DRY_RUN=1` runs the checks and prints the accelerate command without training; `ZERO=2` switches to ZeRO-2;
`NNODES`/`NODE_RANK`/`MASTER_ADDR`/`MASTER_PORT` on every node make it multi-node (node 0 publishes one
`RUN_ID` to the others). The script exports the runtime env itself (`HF_HUB_OFFLINE`, `TORCH_COMPILE_DISABLE`,
`TORCHDYNAMO_DISABLE`, `expandable_segments`).

Key overrides:

| Override | Meaning (defaults in the task config) |
|---|---|
| `batch_size` | per-GPU batch (`robotwin_wam` 2, `robotwin_hybrid` 4) |
| `max_steps` | total optimizer steps (200k / 15k) |
| `save_every` | checkpoint interval (ckpts land in `runs/<task>/<RUN_ID>/checkpoints/weights/step_*.pt`) |
| `eval_every` | held-out val interval (configs default `999999` = off; set `1000` to log `eval/val_loss{,_video,_action,_code}` + `val_code_acc`) |
| `gradient_accumulation_steps` | effective-batch knob (not a speed knob) |
| `learning_rate` / `lr_scheduler_type` | default `1e-4` / `cosine` |
| `save_optimizer_state` | default `false` — weights-only ckpts (optimizer state is ~90 GB/ckpt) |
| `num_epochs` | **must be non-null** even with `max_steps` (trainer reads `int(cfg.num_epochs)`) |
| `wandb.enabled` | on by default (`configs/train.yaml`, project `agenticwam`) |
| `foundation_ckpt` | the pretrained checkpoint the run starts from |


---

## 5. Evaluation — the RoboTwin benchmark

RoboTwin's own `script/eval_policy.py` (SAPIEN, unchanged) runs each episode and calls the policy in
`robotwin/deploy/`. One process evaluates one task in one setting on one GPU; the benchmark runner splits the
whole benchmark into such processes and schedules them. Kill leftover sim processes before starting (§8).

### 5.1 Benchmark a checkpoint — `robotwin/eval/benchmark.py`

Evaluation builds the model from the training config, so `configs/paths/default.yaml` must name the weights
(`wan_vae`, `text_encoder`, `code_model`, `code_vision`; [INSTALL.md](INSTALL.md) §3) — or pass another paths file with
`--launch-args --extra paths=<name>`.

```bash
export AGENTICWAM_HYBRID_SITE=configs/robotwin/site.yaml        # the default; set only for another site file
python robotwin/eval/benchmark.py --model robotwin_wam \
    --ckpt runs/robotwin_wam/<RUN_ID>/checkpoints/weights/step_200000.pt        # 50 tasks x Easy/Hard, 100 episodes each, 8 GPUs
python robotwin/eval/benchmark.py --model <preset> --ckpt <step.pt> --plan smoke --tasks adjust_bottle --protocols clean
```

* **What** — the tasks of `--suite` (`robotwin` = all 50 RoboTwin 2.0 tasks, the default; or a subset with `--tasks a,b`), in
  each of `--protocols` (`clean` = Easy, `demo_clean` scenes; `randomized` = Hard, domain-randomized scenes).
* **How many** — `--plan` from `configs/robotwin/benchmark.yaml`:

  | plan | Easy | Hard | shards | note |
  |---|---|---|---|---|
  | `standard` (default) | 100 | 100 | 10, seeds 0–9 | the RoboTwin 2.0 evaluation protocol |
  | `smoke` | 1 | 1 | 1 | checks the pipeline, not the policy |

  `--episodes` / `--shards` override a plan. Shard k uses RoboTwin seed `seed_base + k`; RoboTwin walks scene seeds
  from `100000 × (1 + seed)`, so shards never share a scene, and the same plan always evaluates the same scenes.
* **Where** — `--gpus 0,1,...` (default all 8), `--per-gpu` jobs on each. Each job is one
  `robotwin/hybrid_eval/launch.py` arm, which exports the environment from `site.yaml` and the preset and writes
  `run_meta.json` (settings + result) beside its rollouts.
* **Resume** — a job whose `run_meta.json` has a result is skipped, so rerunning the same command continues an
  interrupted benchmark; a failed job is retried `--retries` times (default 1). Rollout videos are off unless
  `--render`.
* **Output** — `evaluate_results/robotwin_benchmark/<name>/`: `manifest.json` (every job, its seed and results
  directory), `logs/<protocol>_<task>_s<seed>.log`, and the report (§5.2). `<name>` defaults to
  `<preset>_<checkpoint stem>`. `--dry-run` prints the job count and the first commands.
* **Hybrid** — `--mode hybrid` runs the same benchmark through the code channel (§6); it needs a preset with task
  assets and the vision servers already running.

A full run is 50 tasks × (100 + 100) episodes = 10,000 episodes in 1,000 jobs.

### 5.2 Reports — `robotwin/eval/report.py`

```bash
python robotwin/eval/report.py <name>                       # one run: writes report.md / report.csv / summary.json
python robotwin/eval/report.py <name_a> <name_b> --out compare.md               # several runs side by side
```

Per task and setting: successes / episodes over the finished shards. The average row is the mean of the per-task
success rates (how RoboTwin 2.0 reports), followed by the pooled rate and the episode count. A cell with missing shards is marked `*`. The benchmark runner writes the single-run report when it finishes.

### 5.3 Model presets — `configs/robotwin/models/<name>.yaml`

A preset describes how to evaluate one checkpoint family; `--model` takes a name or a yaml path.

| key | meaning |
|---|---|
| `run` | the training run directory (`--step N` resolves `<run>/checkpoints/weights/step_N.pt`; `--ckpt` overrides) |
| `eval_task` | the task config that builds the model at eval time |
| `action_space`, `action_dim`, `proprio_dim` | `eef64` + 64, or `""` + 14 (§5.5) |
| `use_code_expert` | `false` for a 2-expert checkpoint (default `true`) |
| `code_model` | the SDAR code body when the code expert is on |
| `nis`, `replan` | denoising steps; actions executed per replan |
| `dataset_stats` | normalisation stats (else `site.yaml: dataset_stats`) |
| `task_assets` | `true` (default): the code prompt comes from `configs/robotwin/tasks/<task>/` (models trained on `codegen/sim` data; required for hybrid). `false`: no task assets are used (models trained on the RoboTwin 2.0 dataset's caches) |
| `gen_tokens`, `no_history` | code canvas and history mode (task-asset models) |
| `env` | environment exported into the eval process: instruction maps, skill prompts, decode knobs |

Shipped presets:

| preset | checkpoint family |
|---|---|
| `robotwin_wam` | `robotwin_wam`; `run` = `robotwin_wam_run` in site.yaml |
| `robotwin_hybrid` | `robotwin_hybrid` (hybrid-capable) |

Their settings reproduce the runs' final RoboTwin evaluation (NIS 20, replan 24, instruction map, skill prompt). To
evaluate another checkpoint, pass it with `--ckpt`; for another training config, copy a preset.

### 5.4 One task — `launch.py --mode wam` or `run_task.py`

`benchmark.py` is a scheduler over `launch.py`; to run a single arm by hand:

```bash
python robotwin/hybrid_eval/launch.py --mode wam --model robotwin_wam --ckpt <step.pt> \
    --task beat_block_hammer --proto demo_clean --episodes 7 --seed 1 --shard-gpu 0 [--no-render] [--dry-run]
```

Without `site.yaml` or a preset, call the RoboTwin entry point directly. For a checkpoint with no preset. The hydra keys below are what the entry point forwards to RoboTwin; export the
sim runtime env first.

```bash
export PYTHONPATH="$W/src:${PYTHONPATH:-}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json LD_LIBRARY_PATH="<vulkan lib dir>:${LD_LIBRARY_PATH:-}"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface DIFFSYNTH_MODEL_BASE_PATH="<checkpoints dir>" HF_HOME=<hf_cache>
export AGENTICWAM_ACTION_SPACE=joint                                  # 14-D (5.5)
export AGENTICWAM_CONF_THRESH=0.5 AGENTICWAM_REVEAL_CAP=4              # production decode (5.7)

python robotwin/eval/run_task.py \
  ckpt=runs/robotwin_wam/<RUN_ID>/checkpoints/weights/step_200000.pt gpu_id=0 seed=1 \
  task=robotwin_wam EVALUATION.task_name=handover_block EVALUATION.task_config=demo_clean \
  EVALUATION.eval_num_episodes=7 EVALUATION.dataset_stats_path=$D/dataset_stats.json \
  EVALUATION.num_inference_steps=20 EVALUATION.replan_steps=24 \
  EVALUATION.output_dir=eval_shards/wam_handover_block_g0_s1 \
  model.use_code_expert=true
```

| key | meaning |
|---|---|
| `ckpt` | checkpoint path |
| `task` | the training config of the checkpoint (`robotwin_wam`, `robotwin_hybrid`, or a dataprep config) |
| `EVALUATION.task_name` / `task_config` | RoboTwin task and scene config (`demo_clean` / `demo_randomized`) |
| `EVALUATION.eval_num_episodes` | episodes in this process |
| `EVALUATION.dataset_stats_path` | **norm stats** — must match the action rep (5.5) |
| `EVALUATION.num_inference_steps` | diffusion denoise steps (also the joint-code reveal budget) |
| `EVALUATION.replan_steps` | actions executed per chunk before re-planning (smaller = more reactive) |
| `EVALUATION.output_dir` | **unique per process**, or parallel shards overwrite each other |
| `model.use_code_expert` | `true` = code expert + tri-modal cascade; `false` = 2-expert (video+action only) |
| `model.code_model_name` | code body when the code expert is on |

### 5.5 Action space

Both configs are 14-D joint: `AGENTICWAM_ACTION_SPACE` empty or `joint` (direct qpos) and the 14-D stats
`$D/dataset_stats.json`. (`eef64` installs a cuRobo FK/IK round-trip for a 64-D model; no shipped
RoboTwin config uses it.)

### 5.6 Free-gen vs oracle, and 2-expert baseline
- **Free-gen (default 3-expert):** `model.use_code_expert=true` — the model generates its own code (joint
  co-denoise) each replan, then co-denoises video+action conditioned on it.
- **2-expert baseline:** `model.use_code_expert=false` — no code expert (`infer_action`, video+action only).
- **Oracle (GT-code conditioning):** not wired into this eval path.

### 5.7 Joint co-denoise decode knobs (env, read inside the model)

Read via `os.environ` inside `AgenticWAMCosmos.infer_action_trimodal`
(`src/agenticwam/models/cosmos_mot/agenticwam_model.py`); they reach the sim subprocess because it inherits the
environment, so **export them in the launching shell**.

| Env var | Default | Effect |
|---|---|---|
| `AGENTICWAM_CONF_THRESH` | `0` | `>0` → **adaptive confidence-threshold** reveal: commit a code token only when `max softmax > thresh` |
| `AGENTICWAM_REVEAL_CAP` | `9999` | max code tokens revealed per denoise step (top-confidence kept) |
| `AGENTICWAM_WARMUP` | `0.0` | (linear-reveal path only) fraction of steps before revealing starts |
| `AGENTICWAM_REVEAL_ORDER` | `left2right` | (linear path only) `left2right` or `confidence` |
| `AGENTICWAM_PREFIX_LAST` | `0` | hold the first N code tokens; reveal the body first, prefix commits last |

**Production decode recipe:** `AGENTICWAM_CONF_THRESH=0.5 AGENTICWAM_REVEAL_CAP=4` with 40 inference steps —
adaptive threshold 0.5, at most 4 tokens revealed per step.

---

## 6. Hybrid evaluation — generated code × WAM

`robotwin/hybrid_eval/` runs a RoboTwin evaluation in which, at every replan, the code the model
just generated is either executed on the live environment through the cap-x API (code channel) or ignored in
favour of the WAM action head. The pipeline is task-agnostic and carries no machine-specific paths: the three
things that vary live in `configs/robotwin/`.

### 6.1 Layout

| path | what |
|---|---|
| `configs/robotwin/site.yaml` | machine paths (python env, AgenticWAM / RoboTwin / aspire roots, HF caches, Molmo venv, ffmpeg, vision ports). Git-ignored; copy `site.example.yaml`. |
| `configs/robotwin/models/<name>.yaml` | a checkpoint family + policy settings (run dir, hydra `eval_task`, action space/dims, code model, canvas/replan defaults). Project content. |
| `configs/robotwin/tasks/<task>/`, `tasks_indep/<task>/` | task assets: `preamble.txt` (the prompt: stage-called helper stubs + `# Setup defines:`), `frozen_program.py` (helper implementations = whitelist), `task.yaml` (short name, per-task canvas, provenance, optional `scene.usable_seeds`). Discovered by directory. Which family is used is set by `AGENTICWAM_HYBRID_TASKS`. |
| `robotwin/hybrid_eval/config.py` | loads the three layers (`${key}` references, `${repo}`), env overrides `AGENTICWAM_HYBRID_SITE`, `AGENTICWAM_HYBRID_TASKS`, `CAPX_ROBOTWIN_ROOT`, `AGENTICWAM_ASPIRE_ROOT`. |
| `assemble.py` | `parse_header`, `repair`, `whitelist_ok` (+ `PURE_ALLOWED`), `allowed_names`, `skeleton` + `setup_block` (the prelude). Also an offline assembler CLI. |
| `arbiter.py` | `HybridArbiter`: bind the cap-x API to the live env once per episode, then per replan header -> gates -> repair -> exec, else WAM. Logs `episode{N}_arbitration.jsonl` and `episode{N}_frames.json`. |
| `robotwin/deploy/deploy_policy.py` | the `[hybrid]` hooks (construct under `AGENTICWAM_HYBRID=1`, hand over `_last_code`, `on_replan`, `on_reset`); writes `episode{N}_codes.jsonl` and `episode{N}_result.json`. |
| `vision.py` | `status` / `start --gpu N` / `stop` for Molmo (vLLM), SAM3, Contact-GraspNet. |
| `launch.py` | one eval arm, hybrid or `--mode wam`; builds the environment + hydra overrides itself and writes `run_meta.json` beside the results. |
| `rollout_viewer.py` | `build`: HTML viewer from `run_meta.json` (channel strip, code panel, same-scene pure-WAM video, rescue/loss table); `serve`: static server with HTTP Range (mp4 seeking). §7. |

### 6.2 Task assets — importing frozen programs

Freeze the program in `codegen/sim` (§2.4, §2.7), then (or `scripts/posttrain_annotation_robotwin.sh import`):

    python robotwin/hybrid_eval/import_tasks.py --frozen-root codegen/sim/frozen [--tasks <task>]

It reads `frozen/<task>/task.yaml`, copies the program that file names into `configs/robotwin/tasks/<task>/`,
and refuses it if its sha256 differs from the recorded one. Alongside it go the prompt preamble (`slicing.py
preamble` form, the one the slicing cache embeds for training) and a `task.yaml` (`short`, `gen_tokens` covering
the longest slice in SDAR tokens, the program's sha256, embodiment, instruction mode and the frozen Stage-2
record). Counting `gen_tokens` needs the SDAR tokenizer at `checkpoints/SDAR-1.7B-Chat`.

Nothing else changes: the arbiter reads the step count from the `# [stage k/N]` header, the whitelist from the
frozen program's function definitions plus the preamble's stubs, and the API primitives at bind time.
`--programs-root` still reads the older `programs/<task>/v<N>/` layout.

### 6.3 Which checkpoints

The hybrid channel needs a model trained on the code stream of the same programs it will execute: collect with
`codegen/sim` (§2), build the configs with `robotwin/dataprep` (§3.1), train with `scripts/train_robotwin.sh` (§4). Point a
model preset in `configs/robotwin/models/` at the run (`run`, `eval_task` = the training task config, action
space, dims, code model, canvas). `robotwin_hybrid` is the reference preset and `launch.py`'s default.

### 6.4 Running

    python robotwin/hybrid_eval/launch.py --task beat_block_hammer --model robotwin_hybrid --ckpt <path.pt> \
        --proto demo_randomized --instruction-type seen --episodes 21 --seed 0 \
        --shard-gpu 0 --no-vision --no-render --gen-tokens 124 --replan 24 --tag myrun_beat_block_hammer
    python robotwin/hybrid_eval/launch.py --task beat_block_hammer --model robotwin_hybrid --ckpt <path.pt> \
        --seed 0 --episodes 21 --shard-gpu 0 --mode wam                      # the paired baseline

`--dry-run` prints the resolved plan (tag, checkpoint, assets, canvas, every `AGENTICWAM_*` / `CAPX_*` variable and the
exact command) and starts nothing. `--no-vision` reuses an already-running vision stack; `--no-render` skips the
rollout mp4, which is why per-episode outcomes are written to `episode{N}_result.json` rather than inferred from
video filenames.

Results land in `evaluate_results/robotwin/<ckpt_tag>/<tag>_<task>_g0_s<seed>/<task>/` with `run_meta.json`, the
rollout mp4 (unless `--no-render`), `episode{N}_codes.jsonl`, `episode{N}_arbitration.jsonl`,
`episode{N}_frames.json` and `episode{N}_result.json`. One line per finished arm goes to `site.watch_log`.

**Start the vision stack first** (one per node serves every shard on it):

    PYTHONPATH=robotwin python -m hybrid_eval.vision start --gpu 6 --gpu-seg 7
    PYTHONPATH=robotwin python -m hybrid_eval.vision status

### 6.5 Settings

`launch.py` sets the plumbing itself (`AGENTICWAM_HYBRID`, `AGENTICWAM_HYBRID_TASKS`, `AGENTICWAM_FROZEN_PROGRAM`,
`AGENTICWAM_CODE_PREAMBLE_FILE`, `AGENTICWAM_TRACE`, `AGENTICWAM_INFER_CODE_TOKENS`, ...). Export the following around the command to
change behaviour:

| variable | default | effect |
|---|---|---|
| `AGENTICWAM_NN_REPAIR` | `0` | `1` = when the gate rejects a generated slice, serve the nearest GT slice instead. The code channel is then **not** pure generation, so its share is not comparable with a run that leaves this off. |
| `AGENTICWAM_NN_REPAIR_CORPUS` | derived | root of the slice corpus to match against. Derived from the preset's `eval_task` (task config -> data config -> `code_slices_cache_dir`), so it normally needs no setting. Export it only to match against a different corpus on purpose. |
| `AGENTICWAM_NN_REPAIR_SIM` | `0.80` | minimum similarity of the best match. |
| `AGENTICWAM_NN_REPAIR_MARGIN` | `0.05` | minimum gap between the best and second-best match, i.e. how unambiguous the match must be. |
| `AGENTICWAM_CODE_RETRY_CAP` | `3` | re-executions allowed after code raises — both for a step's slice and for the episode-start prelude. |
| `AGENTICWAM_EVAL_PREPROC` | `parity` | `legacy` \| `parity` \| `parity_crop`. `parity` applies the training pipeline's stage-1 full-frame transform. |
| `AGENTICWAM_EVAL_PREPROC_CROP` | `0.95` | centre-crop fraction, `parity_crop` only. |
| `AGENTICWAM_EOS_RUN_TRUNC` | `0` | `1` = strip the trailing EOS run and drop isolated interior EOS instead of truncating at the FIRST EOS (which turns a stray early EOS into a comment-only slice). |
| `AGENTICWAM_HYBRID_TASKS` | `configs/robotwin/tasks` | which task family to load. A checkpoint trained on a rewritten task set must point this at that set, or every generation is scored against an API it never saw. |
| `AGENTICWAM_FRAME_EVERY` | `12` | sim steps per recorded video frame. |
| `AGENTICWAM_SAVE_ALL_REPLANS` | `0` | `1` = keep a video for every replan, not just the first. |
| `AGENTICWAM_DEPLOY_DTYPE` | `fp32` | policy dtype at inference. |
| `CAPX_EVAL_SEEDS` | unset | explicit seed list instead of the default walk from `100000*(1+seed)`. A task whose `task.yaml` carries `scene.usable_seeds` only spawns its object set on those seeds; shorter lists are cycled. |

### 6.6 What the arbiter does per replan

The slice is accepted for the code channel only if it passes, in order:

1. **header** — `# [stage k/N]`; a step already closed, or below the highest attempted step, goes to WAM.
2. **dependency gate** — every free name must be bound in the namespace, the whitelist or builtins. A slice
   missing a cross-stage input would raise `NameError` only after the replan is spent, so it is refused up front.
3. **vacuous check** — a slice with no executable statement (comments, `pass`, a bare literal) is refused: it
   would "succeed" and close the step without moving the robot.
4. **repeat check** — an identical slice that already failed for this step is skipped without consuming the
   retry budget, unless something that decides the outcome could have moved (the robot executed motion, the
   error was a `RuntimeError`/`ValueError` the world can fix, or another step has closed since).
5. **repair + whitelist** — `repair` comments out non-compiling lines (tier R0/R1/R2); every call must be in the
   whitelist. With `AGENTICWAM_NN_REPAIR=1` a failure here may instead be served from the GT corpus.

Every decision is one record in `episode{N}_arbitration.jsonl`, with `reason` one of `stutter_done`,
`stutter_exhausted`, `stutter_dup`, `missing_deps`, `vacuous`, `gate_fail`, `whitelist`, `unrepairable`,
`nn_repair_step_mismatch`, or `channel: "code"` with `exec_ok`.

---

## 7. Viewing rollouts

`robotwin/hybrid_eval/rollout_viewer.py` turns the results of `launch.py` into a static HTML viewer.
It reads RoboTwin simulation runs only: a result directory needs the `run_meta.json` that `launch.py` writes
(`env: robotwin_sim`; older runs are recognised by their `demo_*` protocol), and anything else is skipped and counted.

    python robotwin/hybrid_eval/rollout_viewer.py build [--results evaluate_results/robotwin] [--out DIR] \
        [--tag-prefix hybrid_] [--model robotwin_hybrid] [--step 15000]
    python robotwin/hybrid_eval/rollout_viewer.py serve [--port 8428] [--dir DIR]

`--out` / `--dir` default to `viewer_out` in `site.yaml`. `build` remuxes each rollout with ffmpeg (`+faststart`, so
the browser can start playing before the whole file arrives; it symlinks when ffmpeg is missing). `serve` answers
HTTP Range requests, which `python -m http.server` does not, so the video timeline is seekable.

**Index page** — one row per (task, protocol, model @ step): hybrid success, the pure-WAM success on the same
scenes, stages driven by code (distinct program stages whose generated code actually issued motion, over the stage
count the program declares), the share of physical time driven by code, and the code share on rescued episodes
(WAM failed, hybrid succeeded). Episode rows are green when hybrid rescued, red when it lost.

**Episode page** — the rollout with a channel strip (blue = code drove, orange = WAM; frame-exact from
`episode{N}_frames.json`, else reconstructed from the arbitration log), the generated code of the replan under the
playhead with its verdict (executed, or the gate's reason), and the pure-WAM rollout of the same scene.

A hybrid run is paired with its pure-WAM baseline by metadata: `meta.baseline` if the arm was launched with
`--baseline <tag>`, otherwise the WAM run with the same (model, step, task, protocol, seed). Episode index N is the
same scene in both because both use the same accepted-seed protocol. A run launched with `--no-render` has no
per-episode outcome; its success cell then shows the shard totals from `run_meta.json`, marked with a dagger.

---

## 8. Notes & gotchas

- **fp32 for the joint co-denoise MoT (LOAD-BEARING).** The joint co-denoise code path must run in
  **fp32** — bf16 crashes the MoT. `deploy_policy.py` builds the policy in fp32 unless
  `AGENTICWAM_DEPLOY_DTYPE=bf16` is set; leave that unset for joint-mode evals.
- **Unique output directory per process.** Parallel shards that share `EVALUATION.output_dir` (or a
  `launch.py --tag`) overwrite each other's results.
- **Kill leftover PIDs before every eval.** SAPIEN sim processes leak GPU memory on crash/interrupt.
  `pkill -f eval_policy.py ; pkill -f run_task.py` and confirm with `nvidia-smi` before relaunching.
- **`AGENTICWAM_ACTION_SPACE` must match the model.** `eef64` for the 64-D model (installs the cuRobo FK/IK
  round-trip adapter, `deploy_policy_eef64.install`); anything else for the 14-D model (direct qpos, no IK).
  Mismatch → wrong action dim / IK failure.
- **Model settings come from the task config, not the CLI.** `deploy_policy._compose_sim_cfg` re-composes the
  task config, so `+model.*` overrides given to the entry point never reach the model build. Declare
  `code_objective` and `code_vision` in the task or model config (both shipped configs do); `CODE_VISION` in the
  environment remains as an ad-hoc fallback. Check the load line: `unexpected=0`.
- **Pass the checkpoint's training config as `task`.** `sim_robotwin.yaml` defaults to `robotwin_wam`.
- **`PYTHONPATH=src` everywhere** (train and eval) — else `import agenticwam` resolves to another checkout.
- **Skill prompt at eval.** Always set `AGENTICWAM_TASK_SKILL_PROMPT`
  (without `AGENTICWAM_SKILL_PROMPT_ONLY`) so the code prompt matches training (`skill_block + wrapped_instruction`);
  omitting it cost 6–11 points.

