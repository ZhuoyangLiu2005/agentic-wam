# AW-0 → RoboDojo stack_bowls post-training bundle

> **In the repo** (`scripts/robodojo_bundle/`) these are the top-level files of the RoboDojo stack_bowls training
> bundle, kept here as their source. They run from the top of an **unpacked bundle**, which has `agentic-wam/` and
> `data/robodojo_x5_stack_bowls/` next to them, not from this folder. See `CHOICES.md`, "Tarball".

Code and data to post-train the AW-0 pretrained checkpoint on the RoboDojo sim task
"Stack the three bowls together." (dual ARX X5 arms with parallel grippers). The AW-0 checkpoint and the frozen
models are **not** in the bundle: `train.sh` reads them from this machine (below).

## Run

```bash
tar -xf robodojo_aw0_stack_bowls.tar && cd robodojo_aw0_stack_bowls
bash setup_env.sh                 # once, only if ~/venvs/agenticwam-data does not exist yet
# only if the weights are not at the defaults (the layout of configs/paths/r6000.yaml):
#   export PRETRAIN_CKPT=.../agenticwam_pretrain.pt WAN_VAE=.../Wan2.2_VAE.pth \
#          CODE_MODEL=.../SDAR-1.7B-Chat-tok CODE_VISION=.../Qwen3-VL-2B-Instruct
DRY_RUN=1 bash train.sh           # preflight: config, paths, code-cache coverage, embeddings, idle GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup bash train.sh > train.log 2>&1 &
```

* GPUs: 8 × 48 GB recommended (global batch 32). One 48 GB GPU runs out of memory at the first optimizer step at
  any batch size; with fewer GPUs set `NPROC` (≥ 4) and expect a smaller global batch.
* W&B: export `WANDB_API_KEY` to log online; otherwise it logs offline (`wandb sync` later).
* Checkpoints: `agentic-wam/runs/posttrain_yam_robodojo_x5_stack_bowls/<RUN_ID>/checkpoints/weights/step_XXXXXX.pt`
  every 5,000 steps; validation loss every 1,000 steps on the held-out episodes 95–99. Pick the checkpoint from the
  validation curve: a small single-task set overfits well before 20,000 steps.
* Extra Hydra overrides go after `train.sh`, e.g. `bash train.sh max_steps=10000`.
* The dry run prints one expected warning: `code cache train: … covers 43,524 frames (offset 0) … cache is longer
  (44,774)`. Train uses episodes 0–94, the cache's first 43,524 frames; val (episodes 95–99) reads the rest through
  `data.val.frame_offset=43524`, which `train.sh` sets. Keep that offset if you change the split.
* `train.sh` writes `agentic-wam/configs/paths/bundle.yaml` from the four weight paths on every launch, so the bundle
  can live anywhere.

## Weights this machine must have

| variable | default | source |
|---|---|---|
| `PRETRAIN_CKPT` | `~/agentic-wam/checkpoints/agentic-wam-pretrain-yamego3k-soup-e1/agenticwam_pretrain.pt` | HF `miniFranka/agentic-wam-pretrain-yamego3k-soup-e1` |
| `WAN_VAE` | `~/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` | HF `Wan-AI/Wan2.2-TI2V-5B` |
| `CODE_MODEL` | `~/models/SDAR-1.7B-Chat-tok` | HF `JetLM/SDAR-1.7B-Chat`: config, tokenizer and modeling code (weights optional, they come from AW-0); `modeling_sdar.py` needs the LossKwargs patch (agentic-wam `CHOICES.md`) |
| `CODE_VISION` | `~/models/Qwen3-VL-2B-Instruct` | HF `Qwen/Qwen3-VL-2B-Instruct` |

`train.sh` stops with a message if one of them is missing.

## Contents

| path | what |
|---|---|
| `train.sh` | the launch command (wraps `agentic-wam/scripts/train_real_yam.sh` with this bundle's data and the weight paths above) |
| `setup_env.sh` | builds the Python 3.10 env from `agentic-wam/configs/env/agenticwam-data.freeze.txt` |
| `agentic-wam/` | the code (repo at the commit this was made from + the RoboDojo changes); `CHOICES.md` has every decision |
| `data/robodojo_x5_stack_bowls/` | the training dataset: LeRobot v2.1 shard (100 episodes, 44,774 frames, 25 Hz, 3 cams 224×320), `dataset_stats_eef64.json`, the instruction embedding, the code-label cache `policy_code_idx_robodojo_v1.pt` (all 100 episodes, 204 codes; `infer_code_tokens` 128) + `episode_order.json`, train/val episode lists |

Not included: the AW-0 checkpoint and the frozen models (see above); Cosmos-Reason1-7B (not needed, the instruction
embedding is precomputed); the raw RoboDojo data; the labeling engine outputs.

## Data facts

* Converted from RoboDojo `lerobot_v2.1` (joint-only) by `agentic-wam/utils/convert_robodojo_v21_to_yam82.py`:
  link6 poses from X5 FK in each arm's base frame (matches RoboDojo's recorded EE poses to ~0.1 mm), YAM 82-D raw
  layout, so the stock YAM transform and recipe apply.
* Cameras: 480×640 → crop rows 16:464 → area-resize to 224×320. Evaluation must apply the same crop.
* fps 25 (the sim's control rate).
