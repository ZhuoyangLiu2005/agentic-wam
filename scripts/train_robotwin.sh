#!/usr/bin/env bash
# Train on RoboTwin simulation data (docs/ROBOTWIN.md; data from codegen/sim -> robotwin/dataprep).
#
#   bash scripts/train_robotwin.sh <task> [hydra overrides ...]
#
# <task> names configs/robotwin/task/<task>.yaml: robotwin_wam (RoboTwin 2.0 official data) or robotwin_hybrid
# (codegen/sim data), both 50 tasks with 14-D joint actions, or a config written by robotwin/dataprep/prepare.py.
# CFG_DROPOUT    export AGENTICWAM_CFG_DROPOUT for tasks trained with stream dropout (unset by default)
#
# Env:  NPROC        GPUs per node (8)              ZERO       DeepSpeed ZeRO stage, 1 or 2 (1)
#       PYTHON       python of the agenticwam env    RUN_ID     reuse a run directory (resume)
#       NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT     multi-node: run the same command on every node
#       WANDB_API_KEY (or WANDB_MODE=offline)       HF_HOME    where the text-encoder weights are cached
#       SKIP_PREFLIGHT=1   skip scripts/preflight_sft.py      DRY_RUN=1   check + print the launch, do not train
set -euo pipefail
TASK="${1:?usage: bash scripts/train_robotwin.sh <task> [hydra overrides ...]}"; shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
NPROC="${NPROC:-8}"
ulimit -c 0 2>/dev/null || true          # a crashed rank can otherwise write a core the size of host memory
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ -n "${CFG_DROPOUT:-}" ]; then export AGENTICWAM_CFG_DROPOUT="$CFG_DROPOUT"; fi   # "p_code,p_video"; set it when the task header asks
if [ "${SKIP_PREFLIGHT:-0}" != 1 ]; then
  "$PY" scripts/preflight_sft.py --task "$TASK" --robot robotwin --nproc "$NPROC" ${DRY_RUN:+--skip-gpu-check} "$@"
fi
exec "$PY" scripts/launch_train.py --task "$TASK" --nproc "$NPROC" --zero "${ZERO:-1}" ${DRY_RUN:+--dry-run} "$@"
