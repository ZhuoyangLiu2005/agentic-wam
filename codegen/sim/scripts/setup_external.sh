#!/usr/bin/env bash
# External source/build/assets and site configuration; activate the intended Python environment first.
set -euo pipefail
SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP_PYTHON="${SETUP_PYTHON:-python}"
usage() {
  cat <<'HELP'
Usage: bash scripts/setup_external.sh COMMAND [options]
  robotwin     Pin RoboTwin, build PyTorch3D/CuRobo and download assets in the active sim environment.
  perception   Pin/install SAM3 and Contact-GraspNet in the active perception environment.
  configure    Write site.yaml using the active sim Python.
               [--robotwin-root PATH] [--trainer-root PATH --train-python PATH]
Conda creation and ordinary pip installation are documented in SETUP.md.
HELP
}
COMMAND="${1:---help}"; [[ $# == 0 ]] || shift
if [[ "$COMMAND" == --help || "$COMMAND" == -h ]]; then usage; exit 0; fi
cd "$SIM_ROOT"
checkout() {
  local path="$1" url="$2" ref="$3" top
  if [[ ! -e "$path" && ! -L "$path" ]]; then
    git clone "$url" "$path"
    git -C "$path" checkout --detach "$ref"
    return
  fi
  top="$(git -C "$path" rev-parse --show-toplevel 2>/dev/null || true)"
  [[ -n "$top" && "$(realpath "$top")" == "$(realpath "$path")" ]] || {
    echo "Existing path is not an independent Git checkout: $path. Left unchanged." >&2; exit 2;
  }
  [[ "$(git -C "$path" rev-parse HEAD)" == "$(git -C "$path" rev-parse "$ref^{commit}")" ]] || {
    echo "Existing checkout differs from required revision: $path ($ref). Left unchanged." >&2; exit 2;
  }
  [[ -z "$(git -C "$path" status --porcelain --untracked-files=no)" ]] || {
    echo "Existing checkout has tracked modifications: $path. Left unchanged." >&2; exit 2;
  }
}
case "$COMMAND" in
  robotwin)
    [[ $# == 0 ]] || { usage >&2; exit 2; }
    command -v nvcc >/dev/null || { echo 'CUDA 12.1 toolkit / nvcc is required; see SETUP.md.' >&2; exit 2; }
    "$SETUP_PYTHON" -c 'import torch; assert torch.version.cuda == "12.1", "Install the documented CUDA 12.1 Torch build first"'
    ROOT="$SIM_ROOT/capsim/third_party/RoboTwin"
    checkout "$ROOT" https://github.com/RoboTwin-Platform/RoboTwin.git bf44be51cf5717a5595ce59447f2cf5263d2aa95
    checkout "$ROOT/envs/curobo" https://github.com/NVlabs/curobo.git v0.7.8
    "$SETUP_PYTHON" -m pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8' --no-build-isolation
    "$SETUP_PYTHON" -m pip install -e "$ROOT/envs/curobo" --no-build-isolation
    if [[ ! -f "$ROOT/assets/.capsim-setup-complete" || ! -d "$ROOT/assets/objects" \
          || ! -d "$ROOT/assets/embodiments" || ! -d "$ROOT/assets/background_texture" ]]; then
      (
        cd "$ROOT/assets"
        "$SETUP_PYTHON" _download.py
        for name in objects embodiments background_texture; do unzip -oq "$name.zip"; done
        cd "$ROOT"
        "$SETUP_PYTHON" script/update_embodiment_config_path.py
        for name in objects embodiments background_texture; do test -d "assets/$name"; done
        touch assets/.capsim-setup-complete
        rm assets/objects.zip assets/embodiments.zip assets/background_texture.zip
      )
    fi
    "$SETUP_PYTHON" -m pip check
    ;;
  perception)
    [[ $# == 0 ]] || { usage >&2; exit 2; }
    checkout "$SIM_ROOT/capsim/third_party/contact_graspnet_pytorch" \
      https://github.com/uynitsuj/contact_graspnet_pytorch.git 8cd98632047e418dc938fc80add258a1ca15f9a9
    checkout "$SIM_ROOT/capsim/third_party/sam3" \
      https://github.com/Max-Fu/sam3.git 6fe87d64a5beb9084923d7a9e002741178635b09
    "$SETUP_PYTHON" -m pip install --no-build-isolation -e "$SIM_ROOT/capsim/third_party/contact_graspnet_pytorch"
    "$SETUP_PYTHON" -m pip install -e "$SIM_ROOT/capsim/third_party/sam3"
    "$SETUP_PYTHON" -m pip check
    ;;
  configure)
    ROBOTWIN_ROOT="$SIM_ROOT/capsim/third_party/RoboTwin"
    TRAINER_PATH=""; DATA_PYTHON=""
    while (($#)); do
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      case "$1" in
        --robotwin-root) ROBOTWIN_ROOT="$2" ;;
        --trainer-root) TRAINER_PATH="$2" ;;
        --train-python) DATA_PYTHON="$2" ;;
        *) echo "Unknown configure option: $1" >&2; exit 2 ;;
      esac
      shift 2
    done
    "$SETUP_PYTHON" - "$SIM_ROOT" "$ROBOTWIN_ROOT" "$TRAINER_PATH" "$DATA_PYTHON" <<'PY'
import os, shutil, sys, tempfile
from pathlib import Path
import yaml
root, robotwin = (Path(p).resolve() for p in sys.argv[1:3])
trainer, train_py = sys.argv[3:5]
if bool(trainer) != bool(train_py):
    sys.exit('Provide both --trainer-root and --train-python, or neither.')
if not (robotwin / 'envs').is_dir():
    sys.exit(f'RoboTwin checkout has no envs/: {robotwin}')
site = root / 'site.yaml'
settings = yaml.safe_load(site.read_text()) if site.exists() else {}
settings = {} if settings is None else settings
if not isinstance(settings, dict):
    sys.exit('site.yaml must contain a mapping.')
settings.update(robotwin_py=str(Path(sys.executable).resolve()), robotwin_root=str(robotwin))
if trainer:
    trainer = Path(trainer).resolve()
    slicer = trainer / 'robotwin/dataprep/slicing.py'
    if not slicer.is_file() or not os.access(train_py, os.X_OK):
        sys.exit('Trainer slicer or data Python is missing; see SETUP.md.')
    settings.update(trainer_root=str(trainer), train_py=str(Path(train_py).resolve()), slicer=str(slicer))
text = yaml.safe_dump(settings, sort_keys=False)
if not site.exists() or site.read_text() != text:
    backup = root / 'local/config/site.yaml.before-setup'
    if site.exists() and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(site, backup)
    fd, temporary = tempfile.mkstemp(dir=root, prefix='site.yaml.')
    try:
        with os.fdopen(fd, 'w') as stream: stream.write(text)
        os.replace(temporary, site)
    finally:
        Path(temporary).unlink(missing_ok=True)
print(f'Configured {site}')
PY
    ;;
  *) echo "Unknown command: $COMMAND" >&2; usage >&2; exit 2 ;;
esac
