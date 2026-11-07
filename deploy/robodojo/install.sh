#!/usr/bin/env bash
# Hook the AgenticWAM adapter into a RoboDojo checkout and give the agentic-wam env the XPolicyLab server deps.
#
#   bash deploy/robodojo/install.sh [ROBODOJO_DIR (default <agentic-wam>/RoboDojo)] [PYTHON (default ~/venvs/agenticwam-data/bin/python)]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RD="$(cd "${1:-$ROOT/RoboDojo}" && pwd)"
PY="${2:-$HOME/venvs/agenticwam-data/bin/python}"
XPL="$RD/XPolicyLab"
for f in setup_policy_server.py utils/process_data.py utils/get_action_dim.sh; do
  [ -e "$XPL/$f" ] || { echo "install.sh: $XPL/$f is missing (incomplete XPolicyLab submodule?):" \
                           "git -C $XPL checkout HEAD -- utils scripts setup_policy_server.py pyproject.toml" >&2; exit 2; }
done
ln -sfn "$ROOT/deploy/robodojo/AgenticWAM" "$XPL/policy/AgenticWAM"
# Optional RoboDojo patch: ROBODOJO_VIDEO_EVERY_STEP=1 records one video frame per control step (real-time videos).
P="$ROOT/deploy/robodojo/patches/robodojo_video_every_step.patch"
if git -C "$RD" apply --reverse --check "$P" 2>/dev/null; then echo "video-every-step patch already applied"
elif git -C "$RD" apply --check "$P" 2>/dev/null; then git -C "$RD" apply "$P" && echo "applied $P"
else echo "install.sh: $P does not apply to this RoboDojo version; eval videos stay one frame per policy call" >&2; fi
echo "linked $XPL/policy/AgenticWAM -> $ROOT/deploy/robodojo/AgenticWAM"
# XPolicyLab's websocket server (client_server/ws) runs inside the policy env.
if command -v uv >/dev/null 2>&1; then uv pip install --python "$PY" "websockets>=14.0" "msgpack>=1.0.8" "msgpack-numpy>=0.4.8" "pydantic>=2.5" "h5py>=3.8"
else "$PY" -m pip install "websockets>=14.0" "msgpack>=1.0.8" "msgpack-numpy>=0.4.8" "pydantic>=2.5" "h5py>=3.8"; fi
"$PY" -c "import websockets, msgpack, msgpack_numpy, pydantic, h5py; print('server deps ok')"
