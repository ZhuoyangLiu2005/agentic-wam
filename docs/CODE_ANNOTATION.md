# Code annotation for a new dataset

How to give a robot dataset **policy-code labels** so AgenticWAM can train its code expert on it:
what the dataset must look like, the two ways to produce labels, and how to wire them into training.

## Where the annotation code lives

```
codegen/
├── sim/        RoboTwin: write and validate programs, then collect demonstrations with per-motion code
│               provenance (ROBOTWIN.md; module setup in codegen/sim/SETUP.md)
├── pretrain/   the VLM labeling engine (pipeline/) and the frozen skill libraries (skill_library/)
└── posttrain/  tools that take a real-robot SFT dataset through labeling: sub-task splits, the bridge
                from engine outputs to a training-order cache, merging, alignment checks, label dumps
```

This document covers **real-robot data**, labeled by the engine in `codegen/pretrain` with the tools in
`codegen/posttrain`. Simulation labels are made differently: `codegen/sim` *executes* a validated program in
RoboTwin and records which line produced each motion, so the code is exact by construction. That pipeline —
collection, training data, training, evaluation and the hybrid evaluation — is [ROBOTWIN.md](ROBOTWIN.md).

**Entry points.** Every command in this document is wrapped by a stage of one script, which fills in the
embodiment and engine paths: `scripts/pretrain_annotation.sh` (the pretraining sources, `yam` / `ego`),
`scripts/posttrain_annotation_yam.sh` (`yam_sft`), `scripts/posttrain_annotation_sharpa.sh` (`tianji_sharpa`); the simulation counterpart,
`scripts/posttrain_annotation_robotwin.sh`, is described in ROBOTWIN.md. Each script's header lists its stages, what they run and which
cost money; e.g. `bash scripts/posttrain_annotation_yam.sh subtasks --dry-run`.

## 1. Two routes, one artifact

| route | what you do | use it when |
|---|---|---|
| **A. VLM labeling pipeline** (`codegen/pretrain/`) | a VLM writes a program for each sub-task slice of each episode, against a declaration-only API and a frozen skill library; programs are checked against the episode's observed gripper/contact events | most datasets — anything with many tasks or varied behaviour. This is how the pretraining sources are labeled. |
| **B. Task program** (your own builder) | you write one canonical program for the task and a script that decides, frame by frame, which part of it is running | a narrow task with fixed structure — branches, counted repetitions, a state that must be stated exactly |

Both end in the same file, a **flat code cache** (§3), referenced from a data config (§6).

## 2. Before you start: the standard format

The dataset must already be in the SFT format — [DATA.md](DATA.md): LeRobot v2.0/v2.1
shards, raw state/action converted to the unified 64-D space by a transform, one instruction and
text embedding per episode, and normalization stats.

Get this right first. A label cache is indexed by frame, so a dataset that is re-converted later
needs its labels rebuilt (or re-aligned, §4.6).

## 3. What you produce: the flat code cache

```python
policy_code_idx_<name>.pt = {
    "codes":          [str, ...],        # deduplicated pool of code strings
    "code_id_by_idx": int32[n_frames],   # global frame index -> index into codes
}
# RobotVideoDataset looks up:  codes[code_id_by_idx[frame_offset + idx]]
```

* **Frame order is the loader's.** Lay the array out in exactly the order `RobotVideoDataset`
  enumerates frames: shard by shard in `dataset_dirs` order, episode by episode within a shard,
  frame by frame. Record that order as `episode_order.json` next to the cache. A cache in any other
  order trains on the wrong frames **with no error**.
* `frame_offset` shifts a contiguous source, validation suffix or node shard into the global index.
* An empty string means "no code for this frame": it is masked out of the loss, not trained as empty.
* The file carries no provenance. Write down which builder and arguments produced it.

## 4. Route A — the VLM labeling pipeline

### 4.1 How it works

Each episode is cut into sub-task slices. For each slice, a VLM actor writes code against a
**declaration-only API** for the embodiment (`codegen/pretrain/pipeline/codegen/api/`). Every API
function declares the gripper/contact events it implies — its *contract* — so a program is scored by
comparing its implied events with those observed in the episode. **Nothing executes.** Recurring
patterns are admitted into a per-embodiment skill library, which is then frozen.

### 4.2 Choose the embodiment

| embodiment | robot | where sub-task slices come from |
|---|---|---|
| `yam` | pretraining source: bimanual arms + parallel grippers | an `annotation.sub_task` column, else `<shard>/subtasks.json` (§4.3), else one slice per episode |
| `ego` | pretraining source: human hands, egocentric video | the same; contact events from the optional fingertip keypoints, a weak signal that is never shown to the actor |
| `yam_sft` | post-training YAM | `<shard>/subtasks.json` (§4.3, gripper splits) |
| `tianji_sharpa` | post-training arm + dexterous hand with tactile sensing | `subtasks.json` (§4.3, tactile splits) |

The two pretraining sources read LeRobot datasets under `yam_root` / `ego_root` of the engine config
(`codegen/pretrain/pipeline/configs/{yam,ego}.yaml`); their format is [PRETRAINING.md](PRETRAINING.md) §1.

**A new robot** needs a new embodiment: a declaration-only API module with contracts
(`codegen/pretrain/pipeline/codegen/api/`), a spec that reads the dataset and names its event source (`codegen/pretrain/pipeline/codegen/embodiments/`),
and registration in the three places that branch on embodiment — `codegen/pretrain/pipeline/codegen/embodiments/__init__.py`,
`codegen/pretrain/pipeline/codegen/api/active.py` and `codegen/pretrain/pipeline/codegen/actor/prompts.py`. Copy the closest existing embodiment. Its skill
library is resolved in `codegen/pretrain/pipeline/codegen/skill_paths.py` (`yam_sft` reuses the `yam` library, `tianji_sharpa` the `ego` one).

**Which view the engine labels.** `yam_sft` reads the dataset named by `YAM_SFT_DATASET` — the training view works;
point it at a higher-resolution copy of the same episodes if the VLM cannot resolve task detail at 224×320.
`tianji_sharpa` reads a **labeling view** named by `TIANJI_DATASET`: the same episodes in the same order as the
training view, packaged for the tactile detector —
* `meta/modality.json` with a `state` block naming slices of `observation.state`: `{left,right}_wrist_eef`
  (pos3 + rot6d; the first three are read as wrist xyz) and per-finger tactile
  `{left,right}_{thumb,index,middle,ring,pinky}_f6` (6 axes each) — grasp events come from these contacts;
* one full-resolution top-camera stream under the video key `top` (the VLM reads it; the policy never does).

### 4.3 Give it sub-task slices

If the dataset has a human `annotation.sub_task` column, the pipeline uses it. Otherwise, generate
`subtasks.json`: a sensor decides **where** a sub-task changes, and a VLM only names **what** happens
in each span.

```bash
P=codegen/pretrain/pipeline
# gripper robots: splits at gripper open/close transitions; writes <shard>/subtasks.json
python codegen/posttrain/annotate_subtasks_gripper.py --dataset <lerobot_dir> --pipeline $P --workers 8
# tactile hands: splits at contact onset/release
python codegen/posttrain/annotate_subtasks_tactile.py --dataset <lerobot_dir> --pipeline $P --out <lerobot_dir>/subtasks.json --workers 8
```

Both take `--dry-run` to inspect the splits before paying for VLM calls.

### 4.4 The skill library

`codegen/pretrain/skill_library/<embodiment>/skills/<kebab-name>/{skill.py, SKILL.md}`, with
`CLAUDE.md` (the format and the admission bar) and `FROZEN.md` (the frozen skill list).

* **Label against a frozen library.** `prompt_version()` hashes the skills doc built from the
  library; changing the library changes the hash and **invalidates every stored label**, forcing paid
  regeneration. To use existing labels, do not touch `skill_library/`.
* **To grow it**, run Stage 1 on a sample, mine candidates (Stage 2), admit skills against the bar in
  `CLAUDE.md` — generic, recurs in ≥2 tasks, grounded, right altitude, not a duplicate — then freeze.
* **Altitude.** A skill must fit inside one sub-task slice. Skills that span slices are never called.
* **Naming.** Skill directories are kebab-case (`brace-and-pick`); code calls snake_case
  (`brace_and_pick(...)`). Convert before matching skills against programs, or usage reads as 0%.

### 4.5 Run it

From `codegen/pretrain/pipeline/`. Set the actor endpoint (`vlm_url`, any OpenAI-compatible chat-completions API) in `configs/<embodiment>.yaml`
and its key in `VLM_API_KEY` (or a `.vlm_api_key` file, chmod 600, git-ignored). The actor model and temperature are in `configs/<embodiment>.yaml`.

```bash
# Stage 1 — actor (PAID): label episodes against the library. Units are "<dataset>:<episode>" lines.
python scripts/run_batch.py --embodiment <emb> --tag <tag> --parallel 5 --tasks-file <units.txt> [--skip-done]

# Stage 2 — mine skill candidates (free); admit, then freeze the library
python scripts/mine_label_patterns.py --embodiment <emb> --tags <tag>

# Stage 3 — corpus scale: one paid call per DISTINCT sub-task label, broadcast to every episode using it
python scripts/skeleton_run.py inventory --embodiment <emb> --units <units.txt> --out inv.json   # free
python scripts/skeleton_run.py generate  --embodiment <emb> --inventory inv.json --parallel 5  # paid
python scripts/skeleton_run.py assemble  --embodiment <emb> --units <units.txt>                # free
python scripts/skeleton_run.py stats

python scripts/gen_progress.py --tag <tag>     # which units are done
```

A small dataset (a few hundred episodes) can stop after Stage 1. Use Stage 3 when many episodes share
sub-task labels — the cost then scales with distinct labels, not episodes.

Outputs land per episode in `outputs/<emb>/<dataset>/<model>/episode_XXX/<tag>/`: `program.py`,
`lines.json`, `chunks.json`, `trace.json`. The label for frame `f` is
`lines.json["frame_src_block"][f] = [a, b]` (1-indexed, inclusive) → `"\n".join(program.py[a-1:b])`.

### 4.6 Turn the outputs into the flat cache

The pipeline often reads a **labeling view** of the dataset, while training reads the **training
view**. Same episodes, but the cache must follow the training view's frame order.

```bash
# one cache for the whole training view
python codegen/posttrain/build_trainable_cache.py --embodiment <yam_sft|tianji_sharpa> --pipeline $P \
    --train-view <lerobot_dir> --task <task> --tag <tag> --work <scratch_dir> --out <lerobot_dir>/policy_code_idx_<name>.pt

# one cache per split, ordered by instantiating the loader from a data config (order correct by construction)
python codegen/posttrain/build_split_caches.py --embodiment <yam_sft|tianji_sharpa> --config configs/real_post_train/data/<robot>.yaml --dataset-root <lerobot_dir> \
    --pipeline $P --task <task> --tag <tag> --name <name> --work <scratch_dir> --out-dir <lerobot_dir>

# concatenate several shard datasets, keeping their caches aligned
python codegen/posttrain/merge_datasets.py --src <dir_a> <dir_b> --cache-name policy_code_idx_<name>.pt --dst <merged_dir>
```

For a pretraining source, `codegen/pretrain/build_code_cache_flat.py` (`scripts/pretrain_annotation.sh flatten`,
[PRETRAINING.md](PRETRAINING.md) §2.3) does the same job and writes `episode_order.json` next to the cache.

### 4.7 Check it

```bash
python codegen/posttrain/verify_code_alignment.py --config configs/real_post_train/data/<robot>.yaml --dataset-root <lerobot_dir> --order <lerobot_dir>/episode_order.json
python codegen/posttrain/dump_code_gt.py --dataset <lerobot_dir> --cache <cache.pt> --out <dump_dir>   # read what the model will see
python codegen/posttrain/verify_rebuild_alignment.py --old <dir_cache_was_built_for> --new <rebuilt_dir>
```

Use only the honest quality metrics from `stats` / `gen_progress`:

| metric | trust |
|---|---|
| `matched_raw_rate` | yes — grounded event match before any repair |
| `repaired_share` | yes — fraction filled in by detector-transcribed repair |
| `matched_slice_rate`, `pin_rate` | **no** — a comment-only program scores 1.000, because tail repair transcribes the detected events |

## 5. Route B — write a task program

For a narrow task you can skip the VLM and write the labels yourself. The builder is a script you
own; this section is the recipe it should follow.

1. **Write the canonical program once**, with the task's real control flow: the branch the operator
   takes (`if bulb lights … else …`), counted loops (`for each item …`), the stopping condition. Use
   the embodiment's API and skill names, so the vocabulary matches Route A labels.
2. **Cut it into stages**, and decide stage boundaries from **sensors and temporal order** — gripper or
   tactile events, kinematic segmentation of the trajectory — **never from appearance**. Two stages that
   look identical on camera (opening vs closing, loosening vs tightening) are told apart by order.
3. **Measure every state the code states.** Object counts, positions, readings, which item is done —
   read them from the episode (a VLM read at chosen frames, or deterministic tracking). Do not copy
   them from the collection label; those are sometimes wrong.
4. **Make each slice self-contained.** A slice restates the goal, the measured current state and what
   is left, so any short training window carries enough context on its own:
   ```python
   # STAGE 2/5 <STAGE NAME> | state: <measured state> | next: <what comes after>
   <the lines of the canonical program for this stage>
   ```
5. **Write the flat cache** (§3) in the training view's order, then run §4.7.

Keep the vocabulary small and consistent: the same situation must produce byte-identical text in every
episode, or the model learns noise.

## 6. Wire it into training

Pass the cache to the train script (`bash scripts/train_real_yam.sh <lerobot_dir> <cache.pt>`), which sets these keys of
the general data config ([DATA.md](DATA.md) §7):

```yaml
use_policy_code: true
policy_code_idx_path: <lerobot_dir>/policy_code_idx_<name>.pt
# optional — a skill-conditioning block per instruction:
skill_prompt_map: <lerobot_dir>/skill_prompt_by_instruction.json
```

* **Skill block** — `codegen/posttrain/build_skill_prompt.py --lib <skill_library/emb> --skills <names> --instruction "<text>" --out <json>`.
* **Language-only ablation** — `codegen/posttrain/build_subgoal_from_code.py --code-cache <cache.pt> --out <subgoal.pt> --write`
  derives a subgoal-text cache with the same frame ids; swapping it in is a path change.
* **`max_code_tokens`** — measure the prompt + code token length over your frames and pick a cap that
  truncates almost none.

How the loader picks a sample's code (`RobotVideoDataset._get`). The branches run in order and the
**last enabled one wins**:

1. `use_policy_code` → on-disk GT by `sha256(task)`, else a template, else `""`. `policy_code_coverage < 1.0`
   withholds a deterministic fraction as `""`; the code stream still runs, so distributed training stays in sync.
2. `skill_labels_path` → a skill id, expanded to code with `obs_grounded_code`.
3. `policy_code_idx_path` → the flat cache (§3).
4. `code_slices` + `cycle_cache_path` → the per-object slice stream used for RoboTwin (see ROBOTWIN.md).

`CodeExpert.build_code_batch` supervises only the code and EOS; the prompt is masked:

```
input_ids = enc("# Instruction: <prompt>\n") + enc(code) + [eos]
labels    = [-100] * len(prompt_ids)          + code_ids + [eos]
```

**`code_canvas` is per checkpoint.** Setting a fixed code canvas (EOS-filled to N tokens) makes
training match the all-`<MASK>` inference canvas — and changes inference too, since
`build_code_batch` runs there. Decide it when you train an arm; switching it later means retraining.

## 7. Pretraining sources

The pretraining sources (`yam`, `ego`) are labeled with Route A at corpus scale (Stage 3) against the frozen
libraries and flattened with `codegen/pretrain/build_code_cache_flat.py`; the commands, in order, are
[PRETRAINING.md](PRETRAINING.md) §2.3.

## 8. Simulation (RoboTwin) labels

RoboTwin labels come from `codegen/sim`: write and validate a task program (Stage A), then execute it to collect
demonstrations whose every motion records the program line that produced it (Stage B). The whole simulation
pipeline is [ROBOTWIN.md](ROBOTWIN.md): data collection, training data and stage slicing, and hybrid evaluation
that executes the same frozen programs.
