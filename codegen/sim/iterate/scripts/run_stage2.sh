#!/usr/bin/env bash
# Stage-2 LOCKED validation for a RoboTwin task — ONE SHOT, never re-run.
# Replays a FROZEN program on 5 clean + 10 randomized locked seeds, CONC jobs across GPUS.
#
# Usage: iterate/scripts/run_stage2.sh <task> [frozen_program.py] [out_root]
#   task     RoboTwin task name
#   program  defaults to $CODEGEN_SIM_AGENT_ROOT/fix_loop/<task>/fix_code.py   (env.sh)
#   out_root defaults to $CODEGEN_SIM_VALIDATION_ROOT/stage2/<task>
# The program is frozen (copied + sha256'd) into out_root, so later edits cannot change what was validated.
set -u
cd "$(dirname "$0")/../.."
source env.sh; codegen_sim_preflight || exit 1
TASK="${1:?usage: run_stage2.sh <task> [program] [out_root]}"
SRC="${2:-$CODEGEN_SIM_AGENT_ROOT/fix_loop/$TASK/fix_code.py}"
OUT="${3:-$CODEGEN_SIM_VALIDATION_ROOT/stage2/$TASK}"
[ -f "$SRC" ] || { echo "no program at $SRC" >&2; exit 1; }
# A default (fix-loop) program at 0/10 debug would pass ~0/15 here: refuse rather than spend the range.
# An explicitly passed program (the evosearch rescue case) is not gated.
if [ -z "${2:-}" ] && [ "${ALLOW_ZERO_DEBUG:-0}" != "1" ]; then
  _dbg=$("$ROBOTWIN_PY" \
           iterate/scripts/gen_progress.py --score-task "$TASK" 2>/dev/null)
  _pass=${_dbg%%/*}; _tot=${_dbg##*/}
  if [ -n "$_pass" ] && [ "${_tot:-0}" -gt 0 ] 2>/dev/null && [ "$_pass" -eq 0 ] 2>/dev/null; then
    echo "ABORT: $TASK scored $_dbg on the debug seeds — Stage 2 would spend the" >&2
    echo "  one-shot locked range confirming a known-zero program." >&2
    echo "  Classify it (STRATEGY -> ../../.claude/skills/robotwin-evosearch/, CAPABILITY -> drop)." >&2
    echo "  To validate anyway: ALLOW_ZERO_DEBUG=1 $0 $TASK" >&2
    exit 1
  fi
fi
# RESUME=1 completes an interrupted run of the SAME (sha256-verified) program: only seeds without a
# trial dir run. To validate a different program, mint a fresh seed range and a new out_root.
RESUME="${RESUME:-0}"
if [ -e "$OUT/oracle" ] && [ "$RESUME" != "1" ]; then
  echo "ABORT: $OUT already has results — Stage 2 is ONE SHOT per program." >&2
  echo "  If that run was KILLED mid-flight and the program is unchanged, finish" >&2
  echo "  it with: RESUME=1 scripts/run_stage2.sh $TASK" >&2
  echo "  To validate a NEW program, mint a fresh seed range (e.g. 301-310/" >&2
  echo "  351-360) and a new out_root; never re-run 201-260." >&2
  exit 1
fi
mkdir -p "$OUT/logs"
PROG="$OUT/frozen_program.py"
if [ "$RESUME" = "1" ]; then
  [ -f "$PROG" ] || { echo "RESUME: no frozen program at $PROG" >&2; exit 1; }
  sha256sum -c "$OUT/frozen_program.sha256" \
    || { echo "RESUME: frozen program changed — refusing to mix two programs' results" >&2; exit 1; }
else
  cp "$SRC" "$PROG"
  sha256sum "$PROG" | tee "$OUT/frozen_program.sha256"
fi
# Stage-marked programs are checked statically at freeze time (frames are checked at collection end).
if grep -qE '^\s*#\s*\[stage' "$PROG"; then
  [ -f "$SLICER" ] || { echo "ABORT: program has stage markers but no slicer at $SLICER (set SLICER=)" >&2; exit 1; }
  python3 "$SLICER" check "$PROG" > "$OUT/stages.txt" 2>&1 \
    || { cat "$OUT/stages.txt" >&2; echo "ABORT: stage contract violated (see $OUT/stages.txt)" >&2; exit 1; }
  echo "stage contract OK: $(tail -2 "$OUT/stages.txt" | head -1)"
else
  echo "WARNING: $PROG has no '# [stage k/N]' markers — it cannot be sliced for code-stream training" >&2
fi
PY="$ROBOTWIN_PY"   # env.sh (preflight already checked it)
# GPUS: pass the coordinator's freed set; otherwise auto-detect usable cards on this host.
MIN_FREE_MIB="${MIN_FREE_MIB:-${CODEGEN_SIM_GPU_MIN_FREE_MIB:-12000}}"   # site.yaml gpu.min_free_mib
if [ -z "${GPUS:-}" ]; then
  GPUS=$(nvidia-smi --query-gpu=index,memory.total,memory.used \
    --format=csv,noheader,nounits 2>/dev/null |
    awk -F', ' -v m="$MIN_FREE_MIB" '($2-$3) >= m {printf "%s ", $1}')
  GPUS="${GPUS:-0}"
  echo "GPUS not set — auto-detected usable GPUs: $GPUS"
fi
_ngpu=$(set -- $GPUS; echo $#)
CONC="${CONC:-$(( _ngpu * ${CODEGEN_SIM_REPLAYS_PER_CARD:-2} ))}"   # site.yaml gpu.replays_per_card
echo "stage2: task=$TASK gpus=[$GPUS] concurrency=$CONC"
# A task whose locked range is spent mints a fresh one (plus a new out_root), e.g.
#   CLEAN_SEEDS="301 302 303 304 305" RAND_SEEDS="351 ... 360"
CLEAN_SEEDS="${CLEAN_SEEDS:-201 202 203 204 205}"
RAND_SEEDS="${RAND_SEEDS:-251 252 253 254 255 256 257 258 259 260}"
echo "stage2 seeds: clean=[$CLEAN_SEEDS] randomized=[$RAND_SEEDS]"
CFG_CLEAN=iterate/configs/priv_clean.yaml
CFG_RAND=iterate/configs/priv_randomized.yaml

gpus=($GPUS); n=${#gpus[@]}; i=0
jobs_list=""
_done_already() {  # $1=seed $2=tag — a completed replay left a trial dir
  ls "$OUT/oracle/$2" 2>/dev/null | grep -q "^trial_$1_"
}
_queue() {  # $1=seed $2=cfg $3=tag
  if [ "$RESUME" = "1" ] && _done_already "$1" "$3"; then
    echo "resume: skipping $3 seed $1 (already has a trial dir)"
    return
  fi
  jobs_list+="$1|$2|$3|${gpus[$((i % n))]}"$'\n'; i=$((i+1))
}
for s in $CLEAN_SEEDS; do _queue "$s" "$CFG_CLEAN" clean; done
for s in $RAND_SEEDS;  do _queue "$s" "$CFG_RAND"  randomized; done
[ -z "$jobs_list" ] && { echo "nothing to run — all seeds already have trials"; }

echo "$jobs_list" | grep -v '^$' | xargs -P "$CONC" -I{} bash -c '
  IFS="|" read -r seed cfg tag gpu <<< "{}"
  CUDA_VISIBLE_DEVICES=$gpu '"$PY"' iterate/scripts/replay_trial.py \
    --code '"$PROG"' --seed "$seed" --config "$cfg" --task '"$TASK"' \
    --output-dir '"$OUT"'/"$tag" > '"$OUT"'/logs/"$tag"_"$seed".log 2>&1
  echo "done seed $seed ($tag, gpu $gpu)"
'
echo "=== STAGE 2 COMPLETE: $TASK ==="
for tag in clean randomized; do
  d="$OUT/oracle/$tag"
  pass=$(ls "$d" 2>/dev/null | grep -c "reward_1.000")
  tot=$(ls "$d" 2>/dev/null | grep -c "trial_")
  echo "$tag: $pass/$tot"
  ls "$d" 2>/dev/null | grep trial_ | sort
done
