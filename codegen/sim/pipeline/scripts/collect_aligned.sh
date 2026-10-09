#!/usr/bin/env bash
# Aligned-dataset collection: replay a FROZEN program over a seed range with the episode saver + code alignment.
#
# Usage:
#   pipeline/scripts/collect_aligned.sh <config.yaml> <start_seed> <end_seed> [gpus...]
#
#   config     a *_collect_*.yaml (oracle_code + save_dir already set)
#   seeds      inclusive range; split evenly across the given GPUs
#   gpus       default: 0
# Collection seeds start at 1001 (clean) / 2001 (randomized); everything below is debug/locked territory.
set -eu
cd "$(dirname "$0")/../.."
source env.sh; codegen_sim_preflight || exit 1
CFG="${1:?usage: collect_aligned.sh <config.yaml> <start> <end> [gpus...]}"
START="${2:?start seed}"; END="${3:?end seed}"; shift 3
GPUS=("${@:-0}"); [ ${#GPUS[@]} -eq 0 ] && GPUS=(0)
[ -f "$CFG" ] || { echo "no such config: $CFG" >&2; exit 1; }

# Seeds < 1001 are refused (debug, locked and minted validation bands); ALLOW_RESERVED=1 lifts this
# for post-freeze official-parity collection only.
if [ "${ALLOW_RESERVED:-0}" != "1" ] && [ "$START" -lt 1001 ]; then
  echo "REFUSING: seeds $START-$END overlap the debug/validation ranges (everything below 1001); collection starts at 1001/2001" >&2
  exit 1
fi

PY="$ROBOTWIN_PY"

# The alignment JSON embeds the program text from this file.
PROG=$(grep -oE "oracle_code: .*" "$CFG" | awk '{print $2}')
[ -f "$PROG" ] || { echo "config's oracle_code does not exist: $PROG" >&2; exit 1; }
export CAPX_ALIGN_CODE_FILE="$PROG"

N=$((END - START + 1)); G=${#GPUS[@]}
CHUNK=$(( (N + G - 1) / G ))
LOGD="$CODEGEN_SIM_LOG_ROOT/collect_$(basename "$CFG" .yaml)"; mkdir -p "$LOGD"   # site.yaml log_root
echo "collect: $N seeds ($START-$END) over ${G} GPU(s), ~$CHUNK each | prog=$PROG"

pids=()
for i in "${!GPUS[@]}"; do
  s=$((START + i * CHUNK)); e=$((s + CHUNK - 1)); [ "$e" -gt "$END" ] && e=$END
  [ "$s" -gt "$END" ] && break
  tmp=$(mktemp "/tmp/collect_XXXX.yaml")
  sed -E "s|^resume_idx:.*|resume_idx: $s|" "$CFG" > "$tmp"
  echo "  gpu ${GPUS[$i]}: seeds $s-$e  (log $LOGD/gpu${GPUS[$i]}.log)"
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" -m capsim.envs.launch \
    --config-path "$tmp" --use-oracle-code True \
    --total-trials "$e" --num-workers 1 \
    > "$LOGD/gpu${GPUS[$i]}.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
SAVE=$(grep -oE "save_dir: .*" "$CFG" | awk '{print $2}')
# The saver writes under $SAVE/{data,alignment,video}/, not flat in $SAVE.
echo "=== done (rc=$rc). $SAVE:"
printf "  episodes  %s\n" "$(find "$SAVE/data" -name '*.hdf5' 2>/dev/null | wc -l)"
printf "  alignment %s\n" "$(ls "$SAVE/alignment" 2>/dev/null | wc -l)"
printf "  video     %s\n" "$(ls "$SAVE/video" 2>/dev/null | wc -l)"
# Stage-marked programs get their slicing cache ($SAVE/slices/stage_frames.json) built here.
if grep -qE '^\s*#\s*\[stage' "$PROG" && [ -f "$SLICER" ] && [ -d "$SAVE/alignment" ]; then
  "${SLICER_PY:-${TRAIN_PY:-python3}}" "$SLICER" build --program "$PROG" --align-dir "$SAVE/alignment" --out "$SAVE/slices" \
    && echo "  slices    $SAVE/slices (stage_frames.json = slicing catalog)" \
    || echo "  slices    FAILED (see above) — fix the stage markers and re-run: $SLICER build --program $PROG --align-dir $SAVE/alignment --out $SAVE/slices"
fi

exit $rc
