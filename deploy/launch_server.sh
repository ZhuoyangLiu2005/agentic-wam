#!/usr/bin/env bash
# Serve a post-trained YAM checkpoint to the robot station (deploy/remote/server.py, token-authenticated).
#
#   CKPT=<step.pt> DATASET=<dataset_dir> INSTR="<instruction>" AGENTICWAM_DEPLOY_TOKEN=<token> bash deploy/launch_server.sh [server args]
#
# DATASET is the dataset directory the checkpoint was post-trained on: it supplies the normalisation stats, so it must
# match training. INFER_CODE_TOKENS (default 64) must match the training run's model.infer_code_tokens.
# Optional env: TASK (default posttrain_yam), GPU (0), PORT (8712), NIS (6 denoising steps), REPLAN (24),
# PYTHON (the agenticwam env's python), HF_HOME. Copy the token to /tmp/agenticwam_token.txt on the station.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${CKPT:?set CKPT to the checkpoint to serve}"
: "${DATASET:?set DATASET to the dataset directory the checkpoint was trained on}"
: "${INSTR:?set INSTR to the task instruction}"
: "${AGENTICWAM_DEPLOY_TOKEN:?set AGENTICWAM_DEPLOY_TOKEN (and copy it to /tmp/agenticwam_token.txt on the station)}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export AGENTICWAM_DEPLOY_DTYPE="${AGENTICWAM_DEPLOY_DTYPE:-bf16}"     # as the shipped server: halves model-load time, no inference cost
export AGENTICWAM_FAST_JOINT="${AGENTICWAM_FAST_JOINT-1}"             # compiled per-step denoise core
export AGENTICWAM_CODE_PIN="${AGENTICWAM_CODE_PIN-1}"                 # keep the step-0 program for the episode
export AGENTICWAM_NOISE_SEED="${AGENTICWAM_NOISE_SEED-1234}"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/.torchinductor_cache_deploy"
CUDA_VISIBLE_DEVICES="${GPU:-0}" exec "${PYTHON:-python}" deploy/remote/server.py \
  --ckpt "$CKPT" --task "${TASK:-posttrain_yam}" --dataset "$DATASET" --override "model.infer_code_tokens=${INFER_CODE_TOKENS:-64}" --instruction "$INSTR" \
  --num-inference-steps "${NIS:-6}" --replan-steps "${REPLAN:-24}" --port "${PORT:-8712}" "$@"
