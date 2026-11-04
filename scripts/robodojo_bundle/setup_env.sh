#!/usr/bin/env bash
# One-time: create the training environment (~/venvs/agenticwam-data by default) from the pinned versions.
# Skip this if that venv already exists on the machine (e.g. the shared home of r6000/d3).
#   bash setup_env.sh [venv_dir]
# Needs: Linux x86_64, NVIDIA driver for CUDA 12.8, internet access to PyPI / download.pytorch.org / GitHub.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${1:-$HOME/venvs/agenticwam-data}"
command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
"$UV" venv --python 3.10 "$VENV"
PY="$VENV/bin/python"
cd "$HERE/agentic-wam"
# exact versions of the environment the recipe was run with (torch 2.7.1+cu128, transformers 4.57.x, deepspeed 0.18.5,
# flash-attn 2.8.3 prebuilt wheel, ...)
"$UV" pip install --python "$PY" --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match -r configs/env/agenticwam-data.freeze.txt
"$UV" pip install --python "$PY" --no-deps -e .
"$PY" -c "import torch, transformers, flash_attn, deepspeed; print(torch.__version__, transformers.__version__, flash_attn.__version__, deepspeed.__version__, 'cuda', torch.cuda.is_available())"
echo "env ready: PYTHON=$PY"
