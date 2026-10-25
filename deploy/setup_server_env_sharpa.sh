#!/usr/bin/env bash
# Environment for the vega + Sharpa deploy server (deploy/launch_server_sharpa.sh). Run on the GPU host.
#
#   bash deploy/setup_server_env_sharpa.sh          # once: venv, packages, frozen weights, SDAR patch, paths file
#   bash deploy/setup_server_env_sharpa.sh check    # imports, CUDA, weights, checkpoint + stats, FK/IK -- loads no model
#
# Env (all optional):
#   VENV       python env to create  ($HOME/venvs/agenticwam-deploy)   PYTHON overrides $VENV/bin/python for check
#   MODELS_DIR frozen weights        ($HOME/models)
#   PATHS_NAME configs/paths/<name>.yaml to write (sharpa_deploy); launch_server_sharpa.sh reads the same name
#   FREEZE=1   install the exact training-env versions (configs/env/agenticwam-data.freeze.txt), not the repo pins
#   CKPT, DATASET  what check looks for (checkpoints/tong-transfer-sharpa/step20000/{model.pt,dataset_stats_eef64.json})
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${VENV:-$HOME/venvs/agenticwam-deploy}"
PY="${PYTHON:-$VENV/bin/python}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
PATHS_NAME="${PATHS_NAME:-sharpa_deploy}"
PATHS_FILE="$ROOT/configs/paths/$PATHS_NAME.yaml"
CKPT="${CKPT:-$ROOT/checkpoints/tong-transfer-sharpa/step20000/model.pt}"
DATASET="${DATASET:-$(dirname "$CKPT")}"
STATS_SHA256=2f8fc4c8198a46910615fbb9e3634cb07d05cf88b06bd18a9e9d0abe31c3a598   # r6000 trex_tong_transfer_aw76_v21

say() { printf '\n== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

setup() {
  command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ] || { say "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; }
  local UV; UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

  say "python env $VENV (Python 3.10)"
  [ -x "$VENV/bin/python" ] || "$UV" venv --python 3.10 "$VENV"
  local VPY="$VENV/bin/python"
  if [ "${FREEZE:-0}" = 1 ]; then   # exact versions of the training env (configs/env/agenticwam-data.freeze.txt)
    "$UV" pip install --python "$VPY" --extra-index-url https://download.pytorch.org/whl/cu128 \
      --index-strategy unsafe-best-match -r configs/env/agenticwam-data.freeze.txt
    "$UV" pip install --python "$VPY" --no-deps -e .
  else                              # repo pins: torch 2.7.1+cu128, transformers 4.57.x, ...
    "$UV" pip install --python "$VPY" --extra-index-url https://download.pytorch.org/whl/cu128 \
      --index-strategy unsafe-best-match -e .
  fi
  # flash-attn: SDAR's RMSNorm needs it (prebuilt wheel: torch 2.7 / cu12 / cxx11abi TRUE / cp310)
  "$UV" pip install --python "$VPY" \
    "flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
  # vega FK/IK bridge (deploy/common/vega_eef64.py -> utils/vega_fk_ik/vega_kinematics.py) + weight download
  "$UV" pip install --python "$VPY" "pin>=3.2" pin-pink qpsolvers daqp "huggingface_hub>=0.30"

  say "frozen weights -> $MODELS_DIR (skips what is already there)"
  HF_HUB_OFFLINE=0 MODELS_DIR="$MODELS_DIR" "$VPY" - <<'EOF'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
m = Path(os.environ["MODELS_DIR"]); m.mkdir(parents=True, exist_ok=True)
jobs = [  # (repo, local dir, allow_patterns, ignore_patterns)
    ("Wan-AI/Wan2.2-TI2V-5B", "Wan2.2-TI2V-5B", ["Wan2.2_VAE.pth"], None),        # frozen video VAE
    ("nvidia/Cosmos-Reason1-7B", "Cosmos-Reason1-7B", None, None),                # instruction encoder
    ("Qwen/Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Instruct", None, None),            # frozen code-vision tower
    # SDAR: config + tokenizer + modeling code only; the checkpoint holds every code-expert tensor
    ("JetLM/SDAR-1.7B-Chat", "SDAR-1.7B-Chat-tok", None, ["*.safetensors", "*.bin", "*.pt"]),
]
for repo, local, allow, ignore in jobs:
    print(f"  {repo} -> {m / local}", flush=True)
    snapshot_download(repo, local_dir=m / local, allow_patterns=allow, ignore_patterns=ignore)
EOF

  say "SDAR modeling_sdar.py: transformers>=4.57 renamed LossKwargs -> TransformersKwargs"
  MODELS_DIR="$MODELS_DIR" "$VPY" - <<'EOF'
import os, shutil
from pathlib import Path
f = Path(os.environ["MODELS_DIR"]) / "SDAR-1.7B-Chat-tok" / "modeling_sdar.py"
src = f.read_text()
old = "from transformers.utils import LossKwargs, auto_docstring, can_return_tuple, is_torch_flex_attn_available, logging\n"
new = ("# " + old + "# transformers>=4.57 renamed LossKwargs -> TransformersKwargs (used only as a TypedDict base below)\n"
       "from transformers.utils import auto_docstring, can_return_tuple, is_torch_flex_attn_available, logging\n"
       "try:\n    from transformers.utils import LossKwargs\nexcept ImportError:\n"
       "    from transformers.utils import TransformersKwargs as LossKwargs\n")
if "TransformersKwargs as LossKwargs" in src:
    print("  already patched")
elif src.count(old) == 1:
    shutil.copy(f, f.with_suffix(".py.orig"))
    f.write_text(src.replace(old, new)); print(f"  patched {f} (original in modeling_sdar.py.orig)")
else:
    raise SystemExit(f"  unexpected modeling_sdar.py at {f}: patch by hand (see CHOICES.md)")
EOF

  if [ ! -f "$PATHS_FILE" ]; then
    say "writing $PATHS_FILE"
    cat > "$PATHS_FILE" <<EOF
# Deploy host weights for launch_server_sharpa.sh (select with paths=$PATHS_NAME). Written by setup.
wan_vae: $MODELS_DIR/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
text_encoder: $MODELS_DIR/Cosmos-Reason1-7B
code_model: $MODELS_DIR/SDAR-1.7B-Chat-tok          # config + tokenizer + modeling code only; weights come from the checkpoint
code_vision: $MODELS_DIR/Qwen3-VL-2B-Instruct
EOF
  fi
  say "setup done. Still needed: the checkpoint at $CKPT and dataset_stats_eef64.json in $DATASET. Then: $0 check"
}

check() {
  local fail=0
  ok() { printf '  ok    %s\n' "$*"; }
  bad() { printf '  FAIL  %s\n' "$*"; fail=1; }
  say "files"
  [ -x "$PY" ] && ok "python $PY" || bad "no python at $PY (run: bash deploy/setup_server_env_sharpa.sh, or set PYTHON)"
  [ -f "$CKPT" ] && ok "checkpoint $CKPT" || bad "no checkpoint at $CKPT (set CKPT)"
  local stats="$DATASET/dataset_stats_eef64.json"
  if [ -f "$stats" ]; then
    local sha; sha="$( (sha256sum "$stats" 2>/dev/null || shasum -a 256 "$stats") | cut -d' ' -f1)"
    [ "$sha" = "$STATS_SHA256" ] && ok "stats $stats (sha256 matches r6000)" \
      || bad "stats $stats sha256 $sha != training stats $STATS_SHA256 (wrong dataset?)"
  else
    bad "no $stats (copy dataset_stats_eef64.json from the training dataset dir; set DATASET)"
  fi
  [ -f "$PATHS_FILE" ] && ok "paths $PATHS_FILE" || bad "no $PATHS_FILE (run: bash deploy/setup_server_env_sharpa.sh)"
  [ "$fail" = 0 ] || die "fix the above first"

  say "python env, weights, FK/IK"
  PATHS_FILE="$PATHS_FILE" "$PY" - <<'EOF' || fail=1
import importlib, os, sys
from pathlib import Path
import yaml
bad = 0
for mod in ("torch", "transformers", "flash_attn", "pinocchio", "pink", "qpsolvers", "daqp", "hydra", "cv2"):
    try:
        m = importlib.import_module(mod); print(f"  ok    import {mod} {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"  FAIL  import {mod}: {e}"); bad = 1
try:
    import torch
    cuda = torch.cuda.is_available()
except ImportError:
    cuda = False
if cuda:
    print(f"  ok    CUDA {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}")
else:
    print("  FAIL  no CUDA (torch missing or torch.cuda.is_available() is False)"); bad = 1
paths = yaml.safe_load(open(os.environ["PATHS_FILE"]))
for key in ("wan_vae", "text_encoder", "code_model", "code_vision"):
    p = Path(paths[key]); print(f"  {'ok  ' if p.exists() else 'FAIL'}  {key}: {p}"); bad |= not p.exists()
sdar = Path(paths["code_model"]) / "modeling_sdar.py"
patched = sdar.exists() and "TransformersKwargs as LossKwargs" in sdar.read_text()
print(f"  {'ok  ' if patched else 'FAIL'}  SDAR LossKwargs patch in {sdar}"); bad |= not patched
sys.exit(bad)
EOF
  # FK -> IK round trip. Not vega_kinematics.py's own __main__: that seeds 0.1 rad off and gives the
  # incremental (teleop-style, gain 0.2) IK only its default 5 steps, so it reports ok=False even when the
  # libraries are fine. 40 steps converge to <0.1 mm when FK/IK work.
  "$PY" -W ignore - <<'EOF' || fail=1
import sys; sys.path.insert(0, "utils/vega_fk_ik")
import numpy as np, vega_kinematics as vk
qL = np.array([0.2, -0.4, 0.3, -0.1, 0.5, 0.0, 0.1]); qR = -qL
TL, TR = vk.fk_both(qL, qR)
ok, a, b = vk.ik_both(TL, TR, qL + 0.1, qR + 0.1, iters=40)
L2, R2 = vk.fk_both(a, b)
err = max(np.linalg.norm(L2[:3, 3] - TL[:3, 3]), np.linalg.norm(R2[:3, 3] - TR[:3, 3])) * 1e3
print(f"  {'ok  ' if ok else 'FAIL'}  vega FK->IK round trip: {err:.3f} mm")
sys.exit(not ok)
EOF
  [ "$fail" = 0 ] && say "all checks passed" || die "some checks failed"
}

case "${1:-setup}" in
  setup) setup ;;
  check) check ;;
  *) die "usage: $0 [setup|check]" ;;
esac
