#!/usr/bin/env bash
# Post-train AW-0 on RoboDojo stack_bowls (dual ARX X5, 25 Hz, YAM 82-D layout). The bundle holds the code and the data;
# the AW-0 checkpoint and the frozen models come from this machine (defaults below = the layout of configs/paths/r6000.yaml).
#
#   bash train.sh [extra hydra overrides ...]
#
# Env:  PYTHON       python of the agenticwam env (default: ~/venvs/agenticwam-data/bin/python if present, else python)
#       NPROC        GPUs per node (8). One 48 GB GPU is NOT enough (optimizer state OOM); use >= 4, ideally 8
#       CUDA_VISIBLE_DEVICES   which GPUs (indices or GPU-<uuid>)
#       RUN_ID       run name; reuse it to resume (with resume=<checkpoint>)
#       WANDB_API_KEY  to log to W&B; without it W&B runs offline (WANDB_MODE=offline)
#       DRY_RUN=1    run the preflight checks and print the launch, do not train
#       MASTER_PORT  set when two runs share a machine
#   Weights on this machine (override any that live elsewhere):
#       PRETRAIN_CKPT  AW-0 agenticwam_pretrain.pt   (HF miniFranka/agentic-wam-pretrain-yamego3k-soup-e1)
#       WAN_VAE        Wan2.2_VAE.pth                 (HF Wan-AI/Wan2.2-TI2V-5B)
#       CODE_MODEL     SDAR-1.7B-Chat dir: config, tokenizer, modeling code with the LossKwargs patch (weights optional)
#       CODE_VISION    Qwen3-VL-2B-Instruct dir      (HF Qwen/Qwen3-VL-2B-Instruct)
# Checkpoints: agentic-wam/runs/posttrain_yam_robodojo_x5_stack_bowls/<RUN_ID>/checkpoints/weights/step_XXXXXX.pt
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V="$HERE/data/robodojo_x5_stack_bowls"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-$HOME/agentic-wam/checkpoints/agentic-wam-pretrain-yamego3k-soup-e1/agenticwam_pretrain.pt}"
WAN_VAE="${WAN_VAE:-$HOME/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"
CODE_MODEL="${CODE_MODEL:-$HOME/models/SDAR-1.7B-Chat-tok}"
CODE_VISION="${CODE_VISION:-$HOME/models/Qwen3-VL-2B-Instruct}"
for v in PRETRAIN_CKPT WAN_VAE CODE_MODEL CODE_VISION; do
  [ -e "${!v}" ] || { echo "train.sh: $v=${!v} does not exist; export $v=<path> (see the header)" >&2; exit 2; }
done
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$HOME/venvs/agenticwam-data/bin/python" ]; then PYTHON="$HOME/venvs/agenticwam-data/bin/python"; else PYTHON=python; fi
fi
export PYTHON
export NPROC="${NPROC:-8}"
export RUN_ID="${RUN_ID:-robodojo_stack_bowls_v1}"
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HOME="${HF_HOME:-$HERE/.hf_home}"                 # nothing is downloaded (offline); keeps caches inside the bundle
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HERE/.hf_datasets_cache}"

cd "$HERE/agentic-wam"
# A paths file with only the keys this run uses (the preflight checks every path in the config). Like
# configs/paths/r6000.yaml.
cat > configs/paths/bundle.yaml <<EOF
# Written by train.sh from PRETRAIN_CKPT / WAN_VAE / CODE_MODEL / CODE_VISION.
wan_vae: $WAN_VAE
code_model: $CODE_MODEL          # config + tokenizer + modeling code; weights come from pretrain_ckpt
code_vision: $CODE_VISION
pretrain_ckpt: $PRETRAIN_CKPT
EOF
exec bash scripts/train_real_yam.sh "$V" "$V/policy_code_idx_robodojo_v1.pt" \
  paths=bundle \
  model.infer_code_tokens=128 \
  data.train.episode_indices="$V/episodes_0_94.json" \
  data.val.episode_indices="$V/episodes_95_99.json" \
  data.val.frame_offset=43524 \
  max_steps=20000 save_every=5000 eval_every=1000 \
  "$@"
