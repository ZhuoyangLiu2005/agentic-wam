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
echo "linked $XPL/policy/AgenticWAM -> $ROOT/deploy/robodojo/AgenticWAM"
# XPolicyLab's websocket server (client_server/ws) runs inside the policy env.
if command -v uv >/dev/null 2>&1; then uv pip install --python "$PY" "websockets>=14.0" "msgpack>=1.0.8" "msgpack-numpy>=0.4.8" "pydantic>=2.5" "h5py>=3.8"
else "$PY" -m pip install "websockets>=14.0" "msgpack>=1.0.8" "msgpack-numpy>=0.4.8" "pydantic>=2.5" "h5py>=3.8"; fi
"$PY" -c "import websockets, msgpack, msgpack_numpy, pydantic, h5py; print('server deps ok')"
