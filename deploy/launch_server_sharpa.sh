#!/usr/bin/env bash
# Serve the vega + Sharpa (tong-transfer) checkpoint to the robot station. Run on the GPU host after
# `bash deploy/setup_server_env_sharpa.sh` and `bash deploy/setup_server_env_sharpa.sh check`.
#
#   bash deploy/launch_server_sharpa.sh [extra deploy/remote/server.py args]   # serve
#   bash deploy/launch_server_sharpa.sh smoke    # full server start with NO checkpoint (actions meaningless)
#
# Everything has a default. Override with env:
#   CKPT       checkpoint            (checkpoints/tong-transfer-sharpa/step20000/model.pt)
#   DATASET    dir holding dataset_stats_eef64.json, the stats the checkpoint was trained with (dir of CKPT)
#   INSTR      task instruction, the dataset's task string (the tong-transfer string below)
#   VENV / PYTHON / PATHS_NAME  as in setup_server_env_sharpa.sh
#   HOST (127.0.0.1: client on this machine; 0.0.0.0 to serve another host)
#   GPU (0)  PORT (8712)  NIS (6 denoise steps)  REPLAN (24 rows per replan)  INFER_CODE_TOKENS (128, as trained)
#   LOG_DIR    server logs + generated code/imagined video (runs/deploy_sharpa)
#   AGENTICWAM_DEPLOY_TOKEN  bearer token; default: read/created at $LOG_DIR/token.txt (run_sharpa_eval.py reads it there)
#
# Runs on the robot station itself; the client is deploy/robot/run_sharpa_eval.py (see its docstring).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${VENV:-$HOME/venvs/agenticwam-deploy}"
PY="${PYTHON:-$VENV/bin/python}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
PATHS_NAME="${PATHS_NAME:-sharpa_deploy}"
PATHS_FILE="$ROOT/configs/paths/$PATHS_NAME.yaml"
CKPT="${CKPT:-$ROOT/checkpoints/tong-transfer-sharpa/step20000/model.pt}"
DATASET="${DATASET:-$(dirname "$CKPT")}"
LOG_DIR="${LOG_DIR:-$ROOT/runs/deploy_sharpa}"
# (own variable: bash 3.2 misparses the apostrophe in "tong's" inside ${INSTR:-...})
DEFAULT_INSTR="There is a metal tong, a coiled small pack of plastic bags, and a red bowl on the table. Use your right hand to pick up the tong, move the tong's end to the pack of plastic bags, close the tong to lift it up, transport it to the red bowl, open the tong to drop the pack of plastic bags in the bowl, and place the tong down on the table."
INSTR="${INSTR:-$DEFAULT_INSTR}"

say() { printf '\n== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

serve() {
  [ -x "$PY" ] || die "no python at $PY (run: bash deploy/setup_server_env_sharpa.sh, or set PYTHON)"
  [ -f "$DATASET/dataset_stats_eef64.json" ] || die "no $DATASET/dataset_stats_eef64.json"
  [ -f "$PATHS_FILE" ] || die "no $PATHS_FILE (run: bash deploy/setup_server_env_sharpa.sh)"
  mkdir -p "$LOG_DIR"
  if [ -z "${AGENTICWAM_DEPLOY_TOKEN:-}" ]; then
    if [ ! -s "$LOG_DIR/token.txt" ]; then
      (umask 077; "$PY" -c "import secrets; print(secrets.token_urlsafe(24))" > "$LOG_DIR/token.txt")
      echo "new deploy token in $LOG_DIR/token.txt (do not commit it)"
    fi
    AGENTICWAM_DEPLOY_TOKEN="$(cat "$LOG_DIR/token.txt")"
  fi
  export AGENTICWAM_DEPLOY_TOKEN
  export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
  export AGENTICWAM_DEPLOY_DTYPE="${AGENTICWAM_DEPLOY_DTYPE:-bf16}"   # halves model-load time, no inference cost
  export AGENTICWAM_FAST_JOINT="${AGENTICWAM_FAST_JOINT-1}"           # compiled per-step denoise core
  export AGENTICWAM_CODE_PIN="${AGENTICWAM_CODE_PIN-1}"               # keep the step-0 program for the episode
  export AGENTICWAM_NOISE_SEED="${AGENTICWAM_NOISE_SEED-1234}"
  export TORCHINDUCTOR_CACHE_DIR="$ROOT/.torchinductor_cache_deploy"
  local log="$LOG_DIR/server_$(date +%Y%m%d_%H%M%S).log"
  echo "serving $CKPT on :${PORT:-8712} (GPU ${GPU:-0}) | log $log | code + imagined video -> $LOG_DIR/videos"
  CUDA_VISIBLE_DEVICES="${GPU:-0}" "$PY" deploy/remote/server.py \
    --ckpt "$CKPT" --task posttrain_sharpa --embodiment sharpa --dataset "$DATASET" \
    --override "paths=$PATHS_NAME" --override "model.infer_code_tokens=${INFER_CODE_TOKENS:-128}" \
    --instruction "$INSTR" --num-inference-steps "${NIS:-6}" --replan-steps "${REPLAN:-24}" \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8712}" --video-log-dir "$LOG_DIR/videos" "$@" 2>&1 | tee "$log"
}

if [ "${1:-}" = smoke ]; then
  shift
  CKPT="$LOG_DIR/__no_checkpoint__.pt" serve "$@"
else
  serve "$@"
fi
