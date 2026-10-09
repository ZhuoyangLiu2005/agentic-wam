#!/usr/bin/env bash
# Code labels for the pretraining sources (yam robot, ego human video) with the VLM labeling engine in
# codegen/pretrain. Walkthrough: docs/PRETRAINING.md §2 and docs/CODE_ANNOTATION.md §4.
#
#   bash scripts/pretrain_annotation.sh <stage> [stage args ...]
#
#   stage      runs                                               cost  what it does
#   subtasks   codegen/posttrain/annotate_subtasks_gripper.py      PAID  yam without sub-task annotation: gripper splits + VLM names
#                                                                        -> <shard>/subtasks.json (--dry-run: splits only, free)
#   label      pipeline/scripts/run_batch.py                      PAID  Stage 1: the actor labels "<dataset>:<episode>" units
#   mine       pipeline/scripts/mine_label_patterns.py            free  Stage 2: skill candidates from Stage-1 labels
#   inventory  pipeline/scripts/skeleton_run.py inventory         free  Stage 3: the distinct sub-task labels
#   generate   pipeline/scripts/skeleton_run.py generate          PAID  Stage 3: one program per distinct label
#   assemble   pipeline/scripts/skeleton_run.py assemble          free  Stage 3: broadcast programs to every episode
#   stats      pipeline/scripts/skeleton_run.py stats             free  quality metrics (matched_raw, repaired_share)
#   progress   pipeline/scripts/gen_progress.py                   free  which units are done
#   flatten    codegen/pretrain/build_code_cache_flat.py          free  engine outputs -> <root>/policy_code_idx.pt + episode_order.json
#   verify     codegen/posttrain/verify_code_alignment.py         free  the cache order matches the training loader
#
# The engine reads its settings from CODEGEN_CONFIG (templates: codegen/pretrain/pipeline/configs/{yam,ego}.yaml;
# set yam_root / ego_root there). Engine stages also take --embodiment yam|ego.
#
# Env:  CODEGEN_CONFIG  configs/yam.yaml or configs/ego.yaml (relative to codegen/pretrain/pipeline)
#       PYTHON          python with codegen/pretrain/pipeline/requirements.txt
#       VLM_API_KEY     key of the actor VLM endpoint (or codegen/pretrain/pipeline/.vlm_api_key, chmod 600) -- PAID stages only
#
# Example (ego):
#   export CODEGEN_CONFIG=configs/ego.yaml                          # ego_root: <ego_root>
#   bash scripts/pretrain_annotation.sh progress --emit-frontier    # units still to do -> outputs/ego/frontier/shard_NN.txt
#   cat codegen/pretrain/pipeline/outputs/ego/frontier/shard_*.txt > units.txt
#   bash scripts/pretrain_annotation.sh inventory --embodiment ego --units units.txt --out inv.json
#   bash scripts/pretrain_annotation.sh generate  --embodiment ego --inventory inv.json --parallel 5
#   bash scripts/pretrain_annotation.sh assemble  --embodiment ego --units units.txt --tag v1
#   bash scripts/pretrain_annotation.sh flatten   --emb ego --dataset-root <ego_root> --tag v1 --out <ego_root>/policy_code_idx.pt
#   bash scripts/pretrain_annotation.sh verify    --config configs/real_pre_train/data/pretrain_magic_soup.yaml \
#        --source ego --dataset-root <ego_root> --order <ego_root>/episode_order.json
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$ROOT/codegen/pretrain/pipeline"
PY="${PYTHON:-python}"
STAGES="subtasks|label|mine|inventory|generate|assemble|stats|progress|flatten|verify"
STAGE="${1:?usage: bash scripts/pretrain_annotation.sh <$STAGES> [args ...]}"; shift

case "$STAGE" in
  subtasks)  exec "$PY" "$ROOT/codegen/posttrain/annotate_subtasks_gripper.py" --pipeline "$ENGINE" "$@" ;;
  label)     cd "$ENGINE" && exec "$PY" scripts/run_batch.py "$@" ;;
  mine)      cd "$ENGINE" && exec "$PY" scripts/mine_label_patterns.py "$@" ;;
  inventory|generate|assemble|stats)
             cd "$ENGINE" && exec "$PY" scripts/skeleton_run.py "$STAGE" "$@" ;;
  progress)  cd "$ENGINE" && exec "$PY" scripts/gen_progress.py "$@" ;;
  flatten)   exec "$PY" "$ROOT/codegen/pretrain/build_code_cache_flat.py" "$@" ;;
  verify)    cd "$ROOT" && exec "$PY" codegen/posttrain/verify_code_alignment.py "$@" ;;
  *) echo "unknown stage '$STAGE' ($STAGES)" >&2; exit 2 ;;
esac
