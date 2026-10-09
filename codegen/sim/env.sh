#!/usr/bin/env bash
# codegen/sim machine contract: `source env.sh`. Values come from site.yaml (from site.example.yaml) via
# scripts/site_env.py; variables already set in the environment win; `codegen_sim_preflight` names what is missing.
#
#   ROBOTWIN_PY                 python with RoboTwin's dependencies (required)       site: robotwin_py
#   CAPX_ROBOTWIN_ROOT          RoboTwin checkout (default: the package symlink)     site: robotwin_root
#   TRAINER_ROOT, TRAIN_PY      downstream trainer repo + its python (Stage B export) site: trainer_root, train_py
#   SLICER                      the stage-contract slicing engine                    site: slicer
#   CODEGEN_SIM_{DATA,AGENT,VALIDATION,LOG}_ROOT   artifact roots                    site: *_root
#   CAPX_ROLLOUT_CACHE          host-local scratch for the saver's per-frame pkls    site: rollout_cache
#   {MOLMO,SAM3,GRASPNET}_SERVICE_URL              vision servers (perception tier)  site: vision.*
#   CODEGEN_SIM_GPU_MIN_FREE_MIB, CODEGEN_SIM_REPLAYS_PER_CARD, CAPSIM_TRIAL_TIMEOUT_S   fleet policy
#   CODEGEN_SIM_FFMPEG_DIR      directory of the ffmpeg binary, prepended to PATH        site: ffmpeg_dir
CODEGEN_SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; export CODEGEN_SIM_ROOT
export PYTHONPATH="$CODEGEN_SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Downstream slicer loads the same placeholder rules used by this collector.
export CAPX_SIM_ROOT="$CODEGEN_SIM_ROOT"
eval "$(python3 "$CODEGEN_SIM_ROOT/scripts/site_env.py" ${CODEGEN_SIM_SITE:+--site "$CODEGEN_SIM_SITE"})"
# ffmpeg (saver video + LeRobot export) is called by name.
[ -n "$CODEGEN_SIM_FFMPEG_DIR" ] && export PATH="$CODEGEN_SIM_FFMPEG_DIR:$PATH"
# Non-machine-specific roots default under outputs/ (git-ignored).
export CODEGEN_SIM_DATA_ROOT="${CODEGEN_SIM_DATA_ROOT:-$CODEGEN_SIM_ROOT/outputs/data}"
export CODEGEN_SIM_AGENT_ROOT="${CODEGEN_SIM_AGENT_ROOT:-$CODEGEN_SIM_ROOT/outputs/agentic}"
export CODEGEN_SIM_VALIDATION_ROOT="${CODEGEN_SIM_VALIDATION_ROOT:-$CODEGEN_SIM_ROOT/outputs/validation}"
export CODEGEN_SIM_LOG_ROOT="${CODEGEN_SIM_LOG_ROOT:-$CODEGEN_SIM_ROOT/outputs/logs}"
codegen_sim_preflight() {
  local ok=0
  python3 "$CODEGEN_SIM_ROOT/scripts/site_env.py" --check ${CODEGEN_SIM_SITE:+--site "$CODEGEN_SIM_SITE"} || ok=1
  [ -n "$ROBOTWIN_PY" ] && [ -x "$ROBOTWIN_PY" ] || { echo "env.sh: ROBOTWIN_PY not executable: '$ROBOTWIN_PY'" >&2; ok=1; }
  [ -n "$CAPX_ROBOTWIN_ROOT" ] && [ -d "$CAPX_ROBOTWIN_ROOT/envs" ] || { echo "env.sh: CAPX_ROBOTWIN_ROOT has no envs/: '$CAPX_ROBOTWIN_ROOT'" >&2; ok=1; }
  [ -n "$SLICER" ] && [ -f "$SLICER" ] || echo "env.sh: note — no slicer at '$SLICER' (stage-contract checks will be skipped)" >&2
  return $ok
}
