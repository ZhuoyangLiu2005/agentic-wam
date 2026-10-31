#!/usr/bin/env bash
# Open-loop server+client replay (scripts/replay_plot.py) for several checkpoints, one after another.
# For each checkpoint: start deploy/launch_server_sharpa.sh, wait for /ping, replay episodes through
# POST /act, plot predicted vs recorded joints, stop the server. NOTHING MOVES: no robot SDK is imported.
#
#   GPU=GPU-<uuid> bash scripts/replay_plot_ckpts.sh checkpoints/.../step5000/model.pt checkpoints/.../step10000/model.pt
#
# Env: GPU (required; on r6000 select by UUID, see memory r6000-gpu-limit)  EPISODES ("0 100 110")  PORT (8712)
#      PY (~/venvs/agenticwam-deploy/bin/python)  OUT_ROOT (runs/replay_plot)
# Each checkpoint dir needs dataset_stats_eef64.json beside model.pt (launch_server_sharpa.sh contract).
# Output: $OUT_ROOT/<run>_<step>/{ep*.npz,replay_compare.png} and runs/deploy_sharpa_<run>_<step>/ (server log, code, videos).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GPU:?set GPU (UUID on r6000)}"
PY="${PY:-$HOME/venvs/agenticwam-deploy/bin/python}"
PORT="${PORT:-8712}"
EPISODES="${EPISODES:-0 100 110}"
OUT_ROOT="${OUT_ROOT:-runs/replay_plot}"

for ckpt in "$@"; do
  d="$(dirname "$ckpt")"; tag="$(basename "$(dirname "$d")")_$(basename "$d")"
  log_dir="runs/deploy_sharpa_$tag"; mkdir -p "$log_dir"
  echo "== $tag: serving $ckpt"
  CKPT="$ckpt" GPU="$GPU" PORT="$PORT" LOG_DIR="$log_dir" bash deploy/launch_server_sharpa.sh > "$log_dir/launch.out" 2>&1 &
  srv=$!
  trap 'pkill -f "remote/server.py --ckpt $ckpt" || true; kill $srv 2>/dev/null || true' EXIT
  until curl -sf "http://127.0.0.1:$PORT/ping" >/dev/null; do
    kill -0 $srv 2>/dev/null || { echo "server died; see $log_dir/launch.out"; tail -20 "$log_dir/launch.out"; exit 1; }
    sleep 10
  done
  echo "server up"
  # shellcheck disable=SC2086
  "$PY" scripts/replay_plot.py --episodes $EPISODES --port "$PORT" \
    --token-file "$log_dir/token.txt" --out "$OUT_ROOT/$tag"
  pkill -f "remote/server.py --ckpt $ckpt" || true; kill $srv 2>/dev/null || true; wait $srv 2>/dev/null || true
  while curl -sf "http://127.0.0.1:$PORT/ping" >/dev/null; do sleep 2; done
  trap - EXIT
done
