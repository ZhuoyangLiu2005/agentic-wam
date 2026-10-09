# Pretraining

AgenticWAM is a tri-modal **video · code · action** World-Action Model on a **Cosmos-Predict2.5-2B** backbone: three
transformer experts (video world model, code-as-policy planner, diffusion action head) share one
Mixture-of-Transformers with per-layer joint self-attention. Pretraining mixes two kinds of data, both mapped into
the unified 64-D action space and both carrying code labels:

| source | what it is | cameras | raw state / action |
|---|---|---|---|
| **yam** | bimanual robot arms with parallel grippers (teleoperated demonstrations) | head + 2 wrist | 82-D |
| **ego** | egocentric video of human hands doing tasks, with wrist poses and hand joints | 1 head camera | 76-D |

This guide is the whole pipeline: the data format of each source, preparing a source (stats, text embeddings, code
labels, checks), the configuration and the launch. Post-training data for a real robot is [DATA.md](DATA.md);
how the code-labeling engine works is [CODE_ANNOTATION.md](CODE_ANNOTATION.md) §4; the environment is
[INSTALL.md](INSTALL.md).

## 0. Overview

```
recordings ──1──> LeRobot shards ──2──> stats ──3──> text embeddings ──4──> code labels ──5──> checks ──6──> pretrain
                  (one root per source)
```

| step | per source | produces | command |
|---|---|---|---|
| 1 | convert recordings | `<root>/train_shard_*/` | your converter (§1) |
| 2a | LeRobot feature stats (v2.0 shards) | `train_shard_*/meta/stats.json` | `python utils/add_lerobot_stats.py --dst <root>` |
| 2b | 64-D normalisation stats | `<root>/dataset_stats_eef64.json` | `PYTHONPATH=src python utils/make_stats.py --transform yam\|ego --dst <root>` |
| 3 | instruction embeddings | `<root>/cosmos_text_embeds_cache/` | `python scripts/precompute_cosmos_text_embeds.py ...` |
| 4 | code labels | `<root>/policy_code_idx.pt`, `<root>/episode_order.json` | `bash scripts/pretrain_annotation.sh <stage>` (§2.3) |
| 5 | checks | — | `verify` stage, `DRY_RUN` of the launch (§2.4) |
| 6 | pretrain | `runs/pretrain_magic_soup/<RUN_ID>/` | `python scripts/launch_train.py --task pretrain_magic_soup ...` (§4) |

A prepared source:

```
<root>/
├── train_shard_0/ … train_shard_N/      one LeRobot v2.0/v2.1 dataset each (§1)
├── dataset_stats_eef64.json             64-D normalisation stats of this source
├── cosmos_text_embeds_cache/            one text embedding per instruction
├── policy_code_idx.pt                   code label of every frame, in the loader's frame order
└── episode_order.json                   that frame order
```

## 1. Data format

Both sources use the LeRobot shard format of [DATA.md](DATA.md) §1 — one 30 Hz clock, 224×320 h264 `yuv420p` videos,
raw absolute state and action, one instruction per episode, shards named `train_shard_0, 1, …`, LeRobot v2.0 or
v2.1. Convert to the 64-D space only through the transforms (at load time), never in the parquet. There is no
converter in this repository: recordings differ per lab; whatever produces the shards must meet the rules below.

### 1.1 yam

Exactly the post-training YAM format ([DATA.md](DATA.md) §1–§2.3): cameras `cam_high`, `cam_left_wrist`,
`cam_right_wrist`; 82-D `observation.state` / `action` with the two EEF poses as row-major 4×4 matrices at `[0:16]`
and `[16:32]` and the grippers at `[32]`, `[33]` (1 open … 0 closed). Transform `YamUnifiedEEFTransform`: EEF and
gripper blocks filled, hand blocks zero.

### 1.2 ego

| feature | dtype | shape | |
|---|---|---|---|
| `observation.images.cam_high` | video | `[224, 320, 3]` | the egocentric (head) camera |
| `observation.state` | float32 | `[76]` | layout below |
| `action` | float32 | `[76]` | same layout: the target the hands reach next — for recorded motion, the state of the next frame |
| `observation.hand_keypoints` | float32 | `[126]` | optional, labeling only: 2 hands × 21 points × xyz (§2.3) |
| `annotation.sub_task` | int64 | `[1]` | optional, labeling only: per-frame sub-task id into `tasks.jsonl` (§2.3) |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | | | as in [DATA.md](DATA.md) §1.1 |

```
[ 0:16]  left wrist pose,  4x4 homogeneous, row-major, metres, a fixed world frame
[16:32]  right wrist pose, 4x4 homogeneous, row-major
[32:54]  left hand, 22 joint angles (radians)
[54:76]  right hand, 22 joint angles (radians)
```

* **Wrist poses in a fixed world frame.** An egocentric camera moves with the head; express the wrists in a frame
  that does not (e.g. compose the camera-frame hand pose with the camera's pose), or the "action" is head motion.
* **Hands retargeted to the 22-joint dexterous hand** of the unified space — the same 22-D convention as the
  Tianji + Sharpa robot, whose raw layout this is (`EgoUnifiedEEFTransform` shares its math). Hand blocks filled,
  gripper slots zero.
* **One camera.** The data config repeats it into the model's three camera slots (`camera_padding: repeat`).
* **Rotations as full matrices**, never a 6-D encoding whose row/column order is implicit.

`observation.hand_keypoints` (if present): 21 points per hand in MediaPipe hand-landmark order (0 wrist, 4 thumb tip,
8 index tip, 12 middle tip, …), left hand first, any consistent frame per hand — only within-hand distances are used.
The training loader ignores both optional columns.

### 1.3 Instructions

Per episode, in `meta/episodes.jsonl` / `meta/tasks.jsonl` exactly as [DATA.md](DATA.md) §1.3. The model sees them
through the template `"A video recorded from a robot's point of view executing the following instruction: {task}"`
(both sources), so embed them with the same script (§2.2).

## 2. Preparing a source

Run every step once per source root.

### 2.1 Stats

```bash
python utils/add_lerobot_stats.py --dst <root>                                  # v2.0 shards: meta/stats.json
PYTHONPATH=src python utils/make_stats.py --transform yam --dst <yam_root>     # -> dataset_stats_eef64.json
PYTHONPATH=src python utils/make_stats.py --transform ego --dst <ego_root>
```

Each source is normalised with its own stats (q01/q99, per step for actions). Blocks a source leaves at zero come out
with `q01 == q99 == 0`; that is expected.

### 2.2 Instruction embeddings

```bash
# instructions.txt: every instruction of the source, one per line
python scripts/precompute_cosmos_text_embeds.py --instructions instructions.txt \
    --out-dir <root>/cosmos_text_embeds_cache [--ckpt <Cosmos-Reason1-7B dir>]
```

About 25 MB per distinct instruction ([DATA.md](DATA.md) §4).

### 2.3 Code labels

Code labels come from the VLM labeling engine in `codegen/pretrain` ([CODE_ANNOTATION.md](CODE_ANNOTATION.md) §4
explains how it works). Every stage is wrapped by `scripts/pretrain_annotation.sh`; its header lists what each stage
runs and which ones cost VLM calls. The engine reads a config file: copy
`codegen/pretrain/pipeline/configs/{yam,ego}.yaml`, set `yam_root` / `ego_root` to the source root, and export
`CODEGEN_CONFIG=configs/<file>.yaml`. Set `vlm_url` there (any OpenAI-compatible chat-completions endpoint) and the key in `VLM_API_KEY`
(or `codegen/pretrain/pipeline/.vlm_api_key`, chmod 600, git-ignored).

**Sub-task slices.** The engine labels an episode one sub-task slice at a time. It takes the slices from, in order:
an `annotation.sub_task` column; a per-shard `subtasks.json`; otherwise the whole episode is one slice. For yam data
without annotation, generate `subtasks.json` from gripper transitions:
`bash scripts/pretrain_annotation.sh subtasks --dataset <yam_root> [--dry-run]`.

**Contact events.** Programs are checked against the events the data shows: gripper open/close transitions for
yam; for ego, a fingertip-aperture detector on `observation.hand_keypoints` (without that column ego episodes carry
no events, and only the annotation and frames guide the labels).

**Skill libraries.** `codegen/pretrain/skill_library/{yam,ego}` are frozen libraries of callable skills shown to the
VLM. Use them as they are, or grow your own (Stage 1 + 2, [CODE_ANNOTATION.md](CODE_ANNOTATION.md) §4.4) before
the corpus-scale run — changing a library invalidates every stored label.

**Corpus-scale labeling (Stage 3)** costs one VLM call per *distinct* sub-task label, then broadcasts the programs to
every episode:

```bash
export CODEGEN_CONFIG=configs/ego.yaml                     # relative to codegen/pretrain/pipeline
bash scripts/pretrain_annotation.sh progress --emit-frontier             # work units -> outputs/ego/frontier/
cat codegen/pretrain/pipeline/outputs/ego/frontier/shard_*.txt > units.txt
bash scripts/pretrain_annotation.sh inventory --embodiment ego --units units.txt --out inv.json   # free
bash scripts/pretrain_annotation.sh generate  --embodiment ego --inventory inv.json --parallel 5  # paid
bash scripts/pretrain_annotation.sh assemble  --embodiment ego --units units.txt --tag v1          # free
bash scripts/pretrain_annotation.sh stats     --embodiment ego
bash scripts/pretrain_annotation.sh flatten   --emb ego --dataset-root <ego_root> --tag v1 \
    --out <ego_root>/policy_code_idx.pt                    # + <ego_root>/episode_order.json
```

A work unit is `<dataset dir>:<episode_index>` (e.g. `train_shard_3:12`); `units.txt` may also be any subset you
choose (the config's `selection` key does the same for every stage). `flatten` writes the label of every frame in
the order the training loader reads the source; an episode without a program gets the empty code, which is masked
out of the loss. Trust `matched_raw_rate` and `repaired_share` from `stats`, not the repaired match rates
([CODE_ANNOTATION.md](CODE_ANNOTATION.md) §4.7).

### 2.4 Checks

```bash
bash scripts/pretrain_annotation.sh verify --config configs/real_pre_train/data/pretrain_magic_soup.yaml \
    --source ego --dataset-root <ego_root> --order <ego_root>/episode_order.json      # cache order == loader order
python utils/verify_instruction_embeds.py --dataset <root>                              # every instruction embedded
python scripts/launch_train.py --task pretrain_magic_soup --dry-run data.yam_root=<yam_root> data.ego_root=<ego_root>
```

A code cache in the wrong frame order trains on the wrong labels with no error — rebuild it (`flatten`) whenever the
source's episodes change.

## 3. Configuration

| file | content |
|---|---|
| `configs/real_pre_train/task/pretrain_magic_soup.yaml` | the recipe: batch 6 per GPU, gradient accumulation 9, lr 1e-4 cosine, one epoch, a checkpoint every 1000 steps (the last kept, with optimizer state, for resuming), `infer_code_tokens: 64` |
| `configs/real_pre_train/data/pretrain_magic_soup.yaml` | the two sources and their mixture |
| `configs/model/agenticwam_cosmos.yaml` | the model: a frozen Qwen3-VL tower + trainable merger feeds a visual prefix to the code expert, joint co-denoised code (SDAR-1.7B body), all experts 28 layers × 16 heads × 128 |

**Initialisation.** The recipe pretrains from scratch: the video expert and tokenizer start from a
Cosmos-Predict2.5-2B video checkpoint (`model.load_pretrain: true`, `paths.cosmos_ckpt`), which is **not
distributed**; the code and action experts start from their language bodies and from random weights. To continue
pretraining on your own sources from the released AW-0 checkpoint instead, start from it and skip the video
checkpoint: `foundation_ckpt=<AW-0 checkpoint> model.load_pretrain=false`. Either way the frozen VAE loads from
`paths.wan_vae`.

The data config takes two roots, `data.yam_root` and `data.ego_root`, and reads everything else from them (shards,
stats, text cache, `policy_code_idx.pt`; `data.{yam,ego}_code_cache` override the cache path). `data.sources.yam` and
`data.sources.ego` are two `RobotVideoDataset`s with their own transform; the training set is their
`ConcatDataset`, wrapped in a `DeadlineDataset` (a slow read retries another index instead of stalling every GPU
in the all-reduce). The mixture is proportional to frame counts; to re-weight, subsample a source's episodes or
add a weighted sampler. Every sample, from either source, has the same contract: `video [3, 3·9, 224, 320]`,
`action [32, 64]`, `proprio [32, 64]`, `prompt` + `prompt_embs`, `policy_code` and the padding masks.

Environment for every launch:

```bash
export PYTHONPATH=src HF_HOME=<hf_cache> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # batch 6 fits; long code samples OOM at 8 without it
export HF_DATASETS_CACHE=<persistent dir>                    # LeRobot builds an arrow cache per shard; keep it, share it across nodes
export AGENTICWAM_CFG_DROPOUT=0.15,0.0                        # the recipe's code / video stream dropout
# WANDB_API_KEY from the environment only (or wandb.enabled=false)
```

## 4. Launch

```bash
# single node, 8 GPUs (smoke: max_steps=3 wandb.enabled=false)
python scripts/launch_train.py --task pretrain_magic_soup --nproc 8 data.yam_root=<yam_root> data.ego_root=<ego_root>

# multi-node: the same command on every node, with the rendezvous in the environment
RUN_ID=<run_name> NNODES=<N> NODE_RANK=<r> MASTER_ADDR=<host of rank 0> MASTER_PORT=<port> \
  python scripts/launch_train.py --task pretrain_magic_soup --nproc 8 data.yam_root=<yam_root> data.ego_root=<ego_root>
```

`--dry-run` prints the resolved accelerate command without starting anything.

### 4.1 The launch stack

```
(one dispatch per node — your cluster's tool)
  └─ python scripts/launch_train.py --task <task> --nproc 8 [overrides]
       └─ accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml scripts/train.py …
```

| env var | maps to (accelerate) | default |
|---|---|---|
| `NNODES` | `--num_machines` | `1` |
| `NODE_RANK` | `--machine_rank` | `0` |
| `MASTER_ADDR` | `--main_process_ip` | `127.0.0.1` |
| `MASTER_PORT` | `--main_process_port` | `29500` |
| `--nproc` | per-node processes; `--num_processes = nproc × NNODES` | 8 |
| `RUN_ID` | run directory `./runs/<task>/<RUN_ID>` | §4.2 |

With `NNODES>1` it passes the full rendezvous plus `--deepspeed_multinode_launcher standard`. Clusters without
inter-node SSH need a per-node dispatcher that starts one `launch_train.py` per node (e.g. a Ray task pinned to each
node). InfiniBand clusters set `NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME` for the fabric.

### 4.2 `RUN_ID`

All ranks must share one run directory. With `NNODES>1` and no `RUN_ID`, node 0 mints a timestamp and publishes it
through a `TCPStore` at `MASTER_ADDR:(MASTER_PORT+11)` (`RUN_ID_SYNC_TIMEOUT`, default 180 s) and the other nodes
read it. Setting `RUN_ID` on every node is simpler, and required to resume into the same directory.

### 4.3 Batch, accumulation, precision

* `batch_size` 6 per GPU with `expandable_segments`; 8 runs out of memory on long code samples.
* `gradient_accumulation_steps` sets the effective batch (`batch × accumulation × world size`); it is not a
  speed knob.
* bf16 mixed precision, DeepSpeed ZeRO-1 (`launch_train.py --zero 2` selects ZeRO-2).

### 4.4 Checkpoints and resume

* Checkpoints go to `./runs/<task>/<RUN_ID>/checkpoints/` every `save_every` steps; `keep_last_ckpts: 1` keeps the
  newest, with optimizer state (`save_optimizer_state: true`).
* `resume=<checkpoint>` continues a run. A weights-only resume must be loaded **before** accelerate/DeepSpeed
  `prepare()`, or the first optimizer step overwrites it with the initial weights.
* The final checkpoint is the `foundation_ckpt` of every post-training recipe
  (`configs/real_post_train/task/posttrain_{yam,sharpa}.yaml`, [REALWORLD.md](REALWORLD.md) §3).
