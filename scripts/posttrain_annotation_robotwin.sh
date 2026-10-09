#!/usr/bin/env bash
# Code-labeled RoboTwin demonstrations: write + validate a task program in simulation (Stage A), collect
# demonstrations whose every motion records the program line that produced it (Stage B), then hand the
# dataset to training and the frozen program to the hybrid evaluation. Walkthrough: docs/ROBOTWIN.md §2-3, §6;
# one-time setup: codegen/sim/SETUP.md (conda envs, RoboTwin checkout, site.yaml).
#
#   bash scripts/posttrain_annotation_robotwin.sh <stage> [stage args ...]
#
#   stage      runs (codegen/sim/ unless noted)               python        what it does
#   baseline   iterate/scripts/run_robotwin_batch.py          $ROBOTWIN_PY  Stage A: a model writes programs with execution feedback (debug seeds)
#   replay     iterate/scripts/replay_trial.py                $ROBOTWIN_PY  Stage A: replay one program on one seed
#   evosearch  iterate/scripts/evosearch_eval.py              $ROBOTWIN_PY  Stage A: score a population of candidate programs
#   validate   iterate/scripts/run_stage2.sh + stage2_report  $ROBOTWIN_PY  Stage A: held-out validation (args: <task> <program.py> <out>)
#   freeze     iterate/scripts/freeze.py                      python3       Stage A: freeze a validated program into frozen/<task>/
#   plan       pipeline/scripts/plan_collection.py            python3       Stage B: how many seeds to attempt
#   collect    pipeline/scripts/make_dataset.py               python3       Stage B: collect + export LeRobot v2.1 + code slices
#   prepare    robotwin/dataprep/prepare.py        $PYTHON       text caches + data/task configs from dataset.yaml (repo root)
#   import     robotwin/hybrid_eval/import_tasks.py $PYTHON     frozen programs -> the hybrid evaluation (repo root)
#
# The fix loop and evolutionary search between `baseline` and `validate` run in Claude Code with the
# repository skills robotwin-fix-loop / robotwin-evosearch (.claude/skills/). Seed bands (§8.3):
# debug 101-105 / 151-155, held-out 201-205 / 251-260, collection from 1001 / 2001.
#
# Env:  codegen/sim stages source codegen/sim/env.sh (ROBOTWIN_PY, CODEGEN_SIM_* roots from site.yaml);
#       run them inside the codegen-sim conda env. PYTHON = the agenticwam trainer env (prepare, import).
#       GPUS = "0 1" for validate.
#
# Example:
#   bash scripts/posttrain_annotation_robotwin.sh validate <task> <program.py> <validation-out>
#   bash scripts/posttrain_annotation_robotwin.sh freeze <task> --program <validation-out>/frozen_program.py \
#        --stage2 <validation-out>/STAGE2_RESULT.md
#   bash scripts/posttrain_annotation_robotwin.sh collect --task <task> --episodes 120 --gpus 0 1 \
#        --task-config demo_clean --start-seed 1001
#   bash scripts/posttrain_annotation_robotwin.sh prepare --datasets <dataset.yaml> --name <name>
#   bash scripts/posttrain_annotation_robotwin.sh import --tasks <task>
#   bash scripts/train_robotwin.sh <name>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM="$ROOT/codegen/sim"
PY="${PYTHON:-python}"
STAGES="baseline|replay|evosearch|validate|freeze|plan|collect|prepare|import"
STAGE="${1:?usage: bash scripts/posttrain_annotation_robotwin.sh <$STAGES> [args ...]}"; shift

sim_env() {
  cd "$SIM"
  set +u                                 # env.sh and its preflight are written for interactive shells
  source ./env.sh
  codegen_sim_preflight
  set -u
}

case "$STAGE" in
  baseline)  sim_env; exec "$ROBOTWIN_PY" iterate/scripts/run_robotwin_batch.py "$@" ;;
  replay)    sim_env; exec "$ROBOTWIN_PY" iterate/scripts/replay_trial.py "$@" ;;
  evosearch) sim_env; exec "$ROBOTWIN_PY" iterate/scripts/evosearch_eval.py "$@" ;;
  validate)  sim_env
             [ $# -ge 3 ] || { echo "usage: validate <task> <program.py> <validation-out>" >&2; exit 2; }
             bash iterate/scripts/run_stage2.sh "$@"
             python3 iterate/scripts/stage2_report.py "$3"
             echo "now write $3/STAGE2_RESULT.md with the score line, then: freeze $1 --program $3/frozen_program.py --stage2 $3/STAGE2_RESULT.md" ;;
  freeze)    sim_env; exec python3 iterate/scripts/freeze.py "$@" ;;
  plan)      sim_env; exec python3 pipeline/scripts/plan_collection.py "$@" ;;
  collect)   sim_env; exec python3 pipeline/scripts/make_dataset.py "$@" ;;
  prepare)   cd "$ROOT"; export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
             exec "$PY" robotwin/dataprep/prepare.py "$@" ;;
  import)    cd "$ROOT"; export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
             exec "$PY" robotwin/hybrid_eval/import_tasks.py --frozen-root "$SIM/frozen" "$@" ;;
  *) echo "unknown stage '$STAGE' ($STAGES)" >&2; exit 2 ;;
esac
