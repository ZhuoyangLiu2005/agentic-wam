# Data preparation for post-training (real robots)

How to turn real-robot recordings into a dataset AgenticWAM can post-train on: the steps in order, what each one
produces, and the format the trainer expects. Other data lives with its pipeline — pretraining (yam robot + ego human video):
[PRETRAINING.md](PRETRAINING.md) §1–2; RoboTwin simulation: [ROBOTWIN.md](ROBOTWIN.md) §2–3.

## 0. Overview

```
recordings ──1──> LeRobot shards ──2──> LeRobot feature stats ──3──> 64-D normalisation stats
                                   ──4──> instruction embeddings ──5──> code labels ──6──> checks ──7──> training
```

| step | produces | command |
|---|---|---|
| 1. convert recordings | `<dataset>/train_shard_*/` (LeRobot v2.0/v2.1) | your converter (§1) |
| 2. LeRobot feature stats (v2.0 only) | `train_shard_*/meta/stats.json` | `python utils/add_lerobot_stats.py --dst <dataset>` |
| 3. normalisation stats | `<dataset>/dataset_stats_eef64.json` | `PYTHONPATH=src python utils/make_stats.py --transform <yam\|tianji_sharpa> --dst <dataset>` |
| 4. instruction embeddings | `<dataset>/cosmos_text_embeds_cache/` | `python scripts/precompute_cosmos_text_embeds.py ...` |
| 5. code labels | `<dataset>/policy_code_idx_<name>.pt`, `episode_order.json` | `scripts/posttrain_annotation_{yam,sharpa}.sh` ([CODE_ANNOTATION.md](CODE_ANNOTATION.md)) |
| 6. checks | — | `utils/verify_instruction_embeds.py`, `codegen/posttrain/verify_code_alignment.py`, `DRY_RUN=1` |
| 7. train | a run | `bash scripts/train_real_{yam,sharpa}.sh <dataset> <code_cache.pt>` |

The finished dataset:

```
<dataset>/
├── train_shard_0/ … train_shard_N/        one LeRobot v2.0/v2.1 dataset each (§1)
│   ├── meta/      info.json  episodes.jsonl  tasks.jsonl  stats.json  (v2.1: episodes_stats.jsonl)
│   ├── data/chunk-000/episode_000000.parquet …
│   ├── videos/chunk-000/observation.images.cam_high/episode_000000.mp4 …
│   │                    observation.images.cam_left_wrist/ …
│   │                    observation.images.cam_right_wrist/ …
│   └── tactile/chunk-000/episode_000000.parquet …   optional extra streams (§2)
├── dataset_stats_eef64.json               normalisation stats in the 64-D space (§3)
├── cosmos_text_embeds_cache/              one text embedding per instruction (§4)
├── episode_order.json                     the loader's frame order (§5)
├── policy_code_idx_<name>.pt              code labels (§5)
├── skill_prompt_by_instruction.json       optional skill block per instruction (§5)
└── source_manifest.json                   optional: the raw recording each episode came from
```

## 1. Convert recordings to LeRobot shards

There is no converter in this repository — recordings differ per robot and cell. Whatever produces the shards must
meet these requirements:

* **One clock.** Resample every stream (state, action, the three cameras, extra sensors) onto one 30 Hz timeline; row
  `i` of an episode's parquet is frame `i` of every video.
* **Videos at the model's size.** h264, `yuv420p`, 30 fps, **224 × 320** (height × width). For example
  `ffmpeg -i in.mp4 -vf scale=320:224 -r 30 -c:v libx264 -pix_fmt yuv420p out.mp4`.
* **Raw, absolute state and action** in one of the supported layouts (§2.3), or a new robot's layout with its own
  transform (§2.4). Do not convert to the 64-D space here; the data config does that at load time.
* **One instruction per episode**, written per episode (§1.3).
* **Shards.** Split however is convenient (one per recording session works well) and name them `train_shard_0`,
  `train_shard_1`, … — the loader reads them in numeric order.
* **LeRobot v2.0 or v2.1, not v3.**

### 1.1 `meta/info.json`

```json
{
  "codebase_version": "v2.0",          // or "v2.1"
  "robot_type": "<robot>",
  "fps": 30,
  "chunks_size": 1000,
  "data_path":  "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
  "total_episodes": 0, "total_frames": 0, "total_tasks": 0, "total_videos": 0, "total_chunks": 1,
  "features": { ... }
}
```

| feature | dtype | shape | notes |
|---|---|---|---|
| `observation.state` | float32 | `[D]` | raw state vector of the robot (§2) |
| `action` | float32 | `[D]` | raw commanded target, same layout as the state (§2) |
| `observation.images.cam_high` | video | `[224, 320, 3]` | head / top camera |
| `observation.images.cam_left_wrist` | video | `[224, 320, 3]` | left wrist camera |
| `observation.images.cam_right_wrist` | video | `[224, 320, 3]` | right wrist camera |
| `timestamp` | float32 | `[1]` | seconds from episode start |
| `frame_index` | int64 | `[1]` | frame within the episode |
| `episode_index` | int64 | `[1]` | episode within the shard |
| `index` | int64 | `[1]` | frame within the shard |
| `task_index` | int64 | `[1]` | row of `tasks.jsonl` this frame's instruction is |

### 1.2 Per-frame data

`data/chunk-000/episode_XXXXXX.parquet` has one row per frame with the non-video features above.

### 1.3 Instructions — `episodes.jsonl`, `tasks.jsonl`

```json
{"episode_index": 0, "tasks": ["<instruction for this episode>"], "length": 1350}
```
```json
{"task_index": 0, "task": "<instruction>"}
```

`tasks.jsonl` lists every distinct instruction in the shard, and every frame's `task_index` must point at its
episode's instruction. If instructions differ per episode, write them per episode — a dataset-wide instruction stamped
onto every episode trains the model on the wrong text (step 6 checks this).

## 2. State and action

### 2.1 Stored raw, converted at load time

Parquet stores each robot's **raw, absolute** vectors. A transform in the data config's processor converts them to the
**unified 64-D** space the model uses, so one action head serves every robot:

```
unified 64-D = [ LEFT 0:32 | RIGHT 32:64 ],   each arm = [ EEF 9 | hand 22 | gripper 1 ]
               EEF = position (3) + rotation 6D (first two COLUMNS of the rotation matrix)

LEFT : eef 0:9    hand 9:31    gripper 31
RIGHT: eef 32:41  hand 41:63   gripper 63
```

| | state | action |
|---|---|---|
| EEF | absolute pose | **delta** from the pose at the chunk's first frame, expressed in that frame |
| hand (22 joints, radians) | absolute | absolute |
| gripper | absolute | absolute target |

A robot fills the blocks it has and leaves the rest zero: a parallel-gripper robot zeroes the 22-D hand block; a
dexterous-hand robot zeroes the gripper slots.

### 2.2 Extra streams and LeRobot feature stats (step 2)

LeRobot keeps statistics of the **raw** features (`{feature: {mean, std, min, max, count}}`), which the loader
aggregates when it builds the dataset. They do not normalise anything — that is §3.

* **v2.0 shards:** `meta/stats.json`, one per shard. The loader **fails without it**
  (`TypeError: 'NoneType' object is not iterable`). Write it with `python utils/add_lerobot_stats.py --dst <dataset>`.
* **v2.1 shards:** `meta/episodes_stats.jsonl`, one row per episode, written by the LeRobot exporter. No `stats.json`.

Extra sensor streams (e.g. tactile) go in a sibling tree such as `tactile/chunk-000/*.parquet`, indexed like `data/`.
The loader does not read them; the code-labeling tools can (tactile sub-task splits).

### 2.3 Supported raw layouts

**Bimanual arms + parallel grippers (YAM) — 82-D** (transform `yam_unified_eef.YamUnifiedEEFTransform`)

| slice | content |
|---|---|
| `[0:16]` | left EEF pose, 4×4 homogeneous, row-major flattened |
| `[16:32]` | right EEF pose, 4×4 homogeneous, row-major flattened |
| `[32]`, `[33]` | left, right gripper |
| `[34:82]` | joint-space and auxiliary fields — not used |

**Bimanual arms + dexterous hands (Tianji + Sharpa) — 76-D** (transform `tianji_sharpa_unified_eef.TianjiSharpaUnifiedEEFTransform`)

| slice | content |
|---|---|
| `[0:16]` | left arm pose, 4×4 homogeneous, row-major flattened |
| `[16:32]` | right arm pose, 4×4 homogeneous, row-major flattened |
| `[32:54]` | left hand, 22 joint angles (radians) |
| `[54:76]` | right hand, 22 joint angles (radians) |

Translations in metres. `action` is the commanded target and `observation.state` the measured one; both absolute.

### 2.4 A new robot

Write a transform in `src/agenticwam/datasets/lerobot/transforms/` that maps your raw vector to the 64-D contract
(forward) and a 64-D delta action back to an absolute target (backward, used at deployment), following the two above,
and a data config next to `configs/real_post_train/data/{yam,sharpa}.yaml` that uses it. Store full rotation matrices or
another unambiguous form — a flattened matrix whose row/column order is unknown is the most common source of a silently
wrong rotation.

## 3. Normalisation stats — `dataset_stats_eef64.json` (step 3)

```bash
PYTHONPATH=src python utils/make_stats.py --transform yam --dst <dataset>             # gripper robot
PYTHONPATH=src python utils/make_stats.py --transform tianji_sharpa --dst <dataset>   # hand robot
```

`--transform` also accepts a dotted class path for a new robot's transform. The stats are computed on the **64-D
transformed** values, over this dataset only:

```
{
  "state":  {"default": {global_{min,max,q01,q99,mean,std}: [64],
                         stepwise_{min,max,q01,q99,mean,std}: [1][64]}},
  "action": {"default": {global_{...}: [64],
                         stepwise_{...}: [32][64]}},      # one row per step of the 32-step chunk
  "num_episodes": N, "num_transition": N
}
```

* The model normalises with `q01`/`q99` (`norm_default_mode: q01/q99`), per step for actions
  (`use_stepwise_action_norm: true`).
* Dimensions a robot leaves at zero come out with `q01 == q99 == 0`. That is expected and handled — do not clamp them
  to a fake range.
* Use the dataset's own stats for training **and** deployment. Stats pooled from other robots are meaningless for the
  blocks this robot actually uses.

## 4. Instruction embeddings — `cosmos_text_embeds_cache/` (step 4)

Every instruction needs a precomputed Cosmos-Reason1 text embedding:

```bash
# instructions.txt: one instruction per line (duplicates are fine)
python scripts/precompute_cosmos_text_embeds.py --instructions instructions.txt \
    --out-dir <dataset>/cosmos_text_embeds_cache [--ckpt <Cosmos-Reason1-7B dir>]
```

```
<sha256(DEFAULT_PROMPT.format(task=instruction))>.cosmos_reason1.pt  ->  {"prompt_embs": [L, 100352] bf16}
```

The script pads/truncates each instruction to `L = 128` tokens (`--max-len`). Each distinct instruction costs about
25 MB.

## 5. Code labels (step 5)

The code expert is supervised by a code label for every frame. How to make them — the VLM labeling pipeline or your own
task program — is [CODE_ANNOTATION.md](CODE_ANNOTATION.md); the wrapper scripts are
`scripts/posttrain_annotation_yam.sh` and `scripts/posttrain_annotation_sharpa.sh`. They produce:

| file | content |
|---|---|
| `policy_code_idx_<name>.pt` | `{"codes": [str], "code_id_by_idx": int32[n_frames]}` — the code label of every frame |
| `episode_order.json` | `[{"trajectory_id": "<shard>:<episode>", "offset": int, "length": int}, ...]` — the order the loader enumerates frames; `code_id_by_idx` is laid out in exactly this order |
| `skill_prompt_by_instruction.json` | `{"<instruction>": "<skill block>"}` — optional skill definitions shown to the code expert |

The loader walks shards in numeric order, episodes in order within a shard, frames in order within an episode;
`offset` is the running frame total. A cache in any other order trains on the wrong frames with no error — step 6
checks it.

## 6. Checks (step 6)

```bash
python utils/verify_instruction_embeds.py --dataset <dataset>          # served instruction == meta, and embedded
python codegen/posttrain/verify_code_alignment.py --config configs/real_post_train/data/<yam|sharpa>.yaml \
    --dataset-root <dataset> --order <dataset>/episode_order.json      # code cache in the loader's order
python codegen/posttrain/dump_code_gt.py --dataset <dataset> --cache policy_code_idx_<name>.pt --out <dir>   # read the labels
DRY_RUN=1 bash scripts/train_real_yam.sh <dataset> <dataset>/policy_code_idx_<name>.pt
```

The dry run composes the training config and runs the preflight — the robot's transform, every path, code-cache
coverage of every frame, every served instruction embedded, idle GPUs — so a missing file or key fails there rather
than an hour into training.

## 7. Train (step 7)

```bash
bash scripts/train_real_yam.sh <dataset> <dataset>/policy_code_idx_<name>.pt [overrides]       # YAM
bash scripts/train_real_sharpa.sh <dataset> <dataset>/policy_code_idx_<name>.pt [overrides]    # Tianji + Sharpa
```

The robot's post-train task config (`configs/real_post_train/task/posttrain_{yam,sharpa}.yaml`) uses one general data
config per robot (`configs/real_post_train/data/{yam,sharpa}.yaml`) that names no dataset: the script sets
`data.dataset_root` and `data.code_cache`, and the config reads the shards (`${lerobot_shards:<dataset>}` lists
`train_shard_*` in numeric order), `dataset_stats_eef64.json` and `cosmos_text_embeds_cache/` from the dataset
directory. Optional overrides: `data.skill_prompt_map=<json>`, and `model.infer_code_tokens=<N>` — the inference code
canvas, default 64; size it to the length of your code labels and serve with the same value.

What the data config resolves to for one dataset (`train`; `val` differs only in `is_training_set`):

```yaml
train:
  _target_: agenticwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset
  dataset_dirs: [<dataset>/train_shard_0, <dataset>/train_shard_1, ...]
  shape_meta:
    images:
      - {key: cam_high,        raw_shape: [3, 224, 320], shape: [3, 224, 320]}
      - {key: cam_left_wrist,  raw_shape: [3, 224, 320], shape: [3, 224, 320]}
      - {key: cam_right_wrist, raw_shape: [3, 224, 320], shape: [3, 224, 320]}
    action: [{key: default, raw_shape: <82 | 76>, shape: 64}]
    state:  [{key: default, raw_shape: <82 | 76>, shape: 64}]
  num_frames: 33                    # a 32-step action chunk + the current frame
  action_video_freq_ratio: 4        # one video frame per 4 action steps -> 9 video frames per sample
  global_sample_stride: 1
  video_size: [224, 320]
  concat_multi_camera: multiview
  context_len: 128
  val_set_proportion: 0.0           # no held-out episodes: validation runs on the training episodes
  is_training_set: true
  pretrained_norm_stats: <dataset>/dataset_stats_eef64.json
  cosmos_text_cache_dir: <dataset>/cosmos_text_embeds_cache
  use_policy_code: true
  policy_code_idx_path: <code_cache.pt>
  policy_code_coverage: 1.0
  skill_prompt_map: null            # or <dataset>/skill_prompt_by_instruction.json
  frame_offset: 0
  processor:
    _target_: agenticwam.datasets.lerobot.processors.agenticwam_processor.AgenticWAMProcessor
    num_output_cameras: 3
    action_output_dim: 64
    proprio_output_dim: 64
    action_state_transforms:
      - {_target_: agenticwam.datasets.lerobot.transforms.<module>.<Transform>, keys: [default]}
    use_stepwise_action_norm: true
    norm_default_mode: q01/q99
    action_state_merger: {_target_: agenticwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign}
    train_transforms: [ToTensor, Resize([224, 320])]
    val_transforms:   [ToTensor, Resize([224, 320])]
```

With `val_set_proportion: 0.0` the validation metrics are training metrics; set it above zero (e.g.
`data.train.val_set_proportion=0.1 data.val.val_set_proportion=0.1`) to hold out episodes. Serving a trained checkpoint
on the robot: [REALWORLD.md](REALWORLD.md) §4.
