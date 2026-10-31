#!/usr/bin/env bash
# Self-contained, run-it-and-leave job: download the run_20k step-15000 checkpoint from Hugging Face, serve it, replay
# episodes through the Sharpa server+client (scripts/replay_plot.py), then write the cross-checkpoint summary.
# Network is only needed for the download (step 1, retried); the server runs with HF_HUB_OFFLINE=1. NOTHING MOVES.
#
#   nohup env GPU=GPU-<uuid> bash scripts/replay_step15000_pipeline.sh > runs/replay_step15000.log 2>&1 &
#
# Env: GPU (required, UUID)  PORT (8713, so it can coexist with another server on 8712)  EPISODES (default all 8)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${GPU:?set GPU (UUID)}"
export PORT="${PORT:-8713}" EPISODES="${EPISODES:-0 100 110 120 130 140 150 160}" OUT_ROOT=runs/replay_plot_more
HF="$HOME/venvs/agenticwam-deploy/bin/hf"
PY="$HOME/venvs/agenticwam-deploy/bin/python"
DIR=checkpoints/tong-transfer-sharpa-run20k
CK=$DIR/run_20k/step15000

echo "== 1/4 download step15000 $(date)"
for i in 1 2 3 4 5 6 7 8 9 10; do
  "$HF" download xinyuan-eval/tong-transfer-sharpa run_20k/step15000/model.pt --local-dir "$DIR" && break
  echo "download attempt $i failed, retrying in 60 s"; sleep 60
done
[ "$(stat -c %s $CK/model.pt 2>/dev/null || echo 0)" = 9959836845 ] || { echo "ERROR: download incomplete"; exit 1; }
cp checkpoints/tong-transfer-sharpa/step20000/{dataset_stats_eef64.json,config.yaml} "$CK/"
echo "download ok $(date)"

echo "== 2/4 serve + replay (episodes: $EPISODES)"
GPU="$GPU" bash scripts/replay_plot_ckpts.sh "$CK/model.pt" || echo "WARNING: replay failed (see runs/deploy_sharpa_run_20k_step15000/launch.out)"
pkill -f "remote/server.py --ckpt $CK" || true

echo "== 3/4 wait for the other replay job (5k/10k/20k on 120-160), if still running"
while pgrep -f "replay_plot_ckpts.sh .*step5000" >/dev/null; do sleep 60; done

echo "== 4/4 summary $(date)"
"$PY" scripts/replay_summary.py
echo "done $(date)"
