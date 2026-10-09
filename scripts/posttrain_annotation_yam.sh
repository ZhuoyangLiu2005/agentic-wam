#!/usr/bin/env bash
# Code annotation of a downstream YAM dataset (bimanual arms + parallel grippers) for SFT.
# Walkthrough: docs/CODE_ANNOTATION.md §2-§6. The dataset must already be in the SFT format (docs/DATA.md).
#
#   bash scripts/posttrain_annotation_yam.sh <stage> [stage args ...]
#
#   stage          runs                                            cost  what it does
#   subtasks       codegen/posttrain/annotate_subtasks_gripper.py  PAID  sub-task slices from gripper open/close transitions (--dry-run: splits only, free)
#   label          pipeline/scripts/run_batch.py                   PAID  the actor labels each slice against the frozen yam library
#   progress       pipeline/scripts/gen_progress.py                free  which units are done
#   cache          codegen/posttrain/build_trainable_cache.py      free  engine outputs -> flat cache in the TRAINING view's order (§4.6)
#   split-caches   codegen/posttrain/build_split_caches.py         free  one cache per split, ordered by the data config's loader
#   merge          codegen/posttrain/merge_datasets.py             free  concatenate shard datasets with their caches
#   verify         codegen/posttrain/verify_code_alignment.py      free  the cache order matches the loader (§4.7)
#   rebuild-check  codegen/posttrain/verify_rebuild_alignment.py   free  a re-converted dataset still lines up with its cache
#   dump           codegen/posttrain/dump_code_gt.py               free  read the labels the model will see
#   skill-prompt   codegen/posttrain/build_skill_prompt.py         free  optional skill-conditioning block per instruction (§6)
#   subgoal        codegen/posttrain/build_subgoal_from_code.py    free  language-only ablation cache with the same frame ids
#
# The engine embodiment is `yam_sft`; cache/split-caches get --embodiment yam_sft and --pipeline filled in.
# Run a stage with --help for its full arguments.
#
# Env:  YAM_SFT_DATASET   the LeRobot view the engine labels (subtasks, label)
#       PYTHON         python with codegen/pretrain/pipeline/requirements.txt
#       VLM_API_KEY     key of the actor VLM endpoint (or codegen/pretrain/pipeline/.vlm_api_key, chmod 600) -- PAID stages only
#
# Example:
#   export YAM_SFT_DATASET=<lerobot_dir>          # shards train_shard_*/ ; subtasks.json is written per shard
#   bash scripts/posttrain_annotation_yam.sh subtasks --workers 8
#   bash scripts/posttrain_annotation_yam.sh label --tag v1 --parallel 5 --tasks-file units.txt
#   bash scripts/posttrain_annotation_yam.sh cache --train-view <lerobot_dir> --task <task> --tag v1 \
#        --work <scratch> --out <lerobot_dir>/policy_code_idx_<name>.pt
#   bash scripts/posttrain_annotation_yam.sh verify --config configs/real_post_train/data/yam.yaml --dataset-root <lerobot_dir> --order <scratch>/episode_order.json
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="$ROOT/codegen/pretrain/pipeline"
POST="$ROOT/codegen/posttrain"
PY="${PYTHON:-python}"
EMB=yam_sft
LIB=yam
STAGES="subtasks|label|progress|cache|split-caches|merge|verify|rebuild-check|dump|skill-prompt|subgoal"
STAGE="${1:?usage: bash scripts/posttrain_annotation_yam.sh <$STAGES> [args ...]}"; shift
export CODEGEN_EMBODIMENT="$EMB"
need_view() { [ -n "${YAM_SFT_DATASET:-}" ] && [ -d "$YAM_SFT_DATASET" ] || { echo "set YAM_SFT_DATASET to the LeRobot view to label" >&2; exit 2; }; }

case "$STAGE" in
  subtasks)  need_view; exec "$PY" "$POST/annotate_subtasks_gripper.py" --dataset "$YAM_SFT_DATASET" --pipeline "$ENGINE" "$@" ;;
  label)     need_view; cd "$ENGINE" && exec "$PY" scripts/run_batch.py --embodiment "$EMB" "$@" ;;
  progress)  cd "$ENGINE" && exec "$PY" scripts/gen_progress.py "$@" ;;
  cache)     exec "$PY" "$POST/build_trainable_cache.py" --embodiment "$EMB" --pipeline "$ENGINE" "$@" ;;
  split-caches)
             exec "$PY" "$POST/build_split_caches.py" --embodiment "$EMB" --pipeline "$ENGINE" "$@" ;;
  merge)     exec "$PY" "$POST/merge_datasets.py" "$@" ;;
  verify)    exec "$PY" "$POST/verify_code_alignment.py" "$@" ;;
  rebuild-check)
             exec "$PY" "$POST/verify_rebuild_alignment.py" "$@" ;;
  dump)      exec "$PY" "$POST/dump_code_gt.py" "$@" ;;
  skill-prompt)
             exec "$PY" "$POST/build_skill_prompt.py" --lib "$ROOT/codegen/pretrain/skill_library/$LIB" "$@" ;;
  subgoal)   exec "$PY" "$POST/build_subgoal_from_code.py" "$@" ;;
  *) echo "unknown stage '$STAGE' ($STAGES)" >&2; exit 2 ;;
esac
