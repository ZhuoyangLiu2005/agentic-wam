#!/usr/bin/env bash
# Downstream SFT on a real-world YAM task (bimanual arms + parallel grippers; 82-D raw -> unified 64-D).
#
#   bash scripts/train_real_yam.sh <dataset_dir> <code_cache.pt> [hydra overrides ...]
#
# Trains the posttrain_yam recipe (configs/real_post_train/task/posttrain_yam.yaml) on one dataset in the SFT format
# (docs/DATA.md): <dataset_dir> holds train_shard_*/, dataset_stats_eef64.json and cosmos_text_embeds_cache/;
# <code_cache.pt> is its flat code-label cache (docs/CODE_ANNOTATION.md). Set the inference code canvas to your
# code length with model.infer_code_tokens=<N> (default 64).
#
# Env:  NPROC        GPUs per node (8)              ZERO       DeepSpeed ZeRO stage, 1 or 2 (1)
#       PYTHON       python of the agenticwam env    RUN_ID     reuse a run directory (resume)
#       NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT     multi-node: run the same command on every node
#       WANDB_API_KEY (or WANDB_MODE=offline)       HF_HOME    where the text-encoder weights are cached
#       SKIP_PREFLIGHT=1   skip scripts/preflight_sft.py      DRY_RUN=1   check + print the launch, do not train
#       CFG_DROPOUT  "p_code,p_video" stream dropout (0.15,0.0, as the shipped runs)
set -euo pipefail
DATASET="${1:?usage: bash scripts/train_real_yam.sh <dataset_dir> <code_cache.pt> [hydra overrides ...]}"
CODE_CACHE="${2:?usage: bash scripts/train_real_yam.sh <dataset_dir> <code_cache.pt> [hydra overrides ...]}"; shift 2
TASK=posttrain_yam
set -- "data.dataset_root=$(cd "$DATASET" && pwd)" "data.code_cache=$CODE_CACHE" "$@"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
NPROC="${NPROC:-8}"
ulimit -c 0 2>/dev/null || true          # a crashed rank can otherwise write a core the size of host memory
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export AGENTICWAM_CFG_DROPOUT="${CFG_DROPOUT:-0.15,0.0}"   # "p_code,p_video": stream dropout for compositional CFG
if [ "${SKIP_PREFLIGHT:-0}" != 1 ]; then
  "$PY" scripts/preflight_sft.py --task "$TASK" --robot yam --nproc "$NPROC" ${DRY_RUN:+--skip-gpu-check} "$@"
fi
exec "$PY" scripts/launch_train.py --task "$TASK" --nproc "$NPROC" --zero "${ZERO:-1}" ${DRY_RUN:+--dry-run} "$@"

