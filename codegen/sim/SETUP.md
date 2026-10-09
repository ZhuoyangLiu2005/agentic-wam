# Setup

All commands run from this module's `codegen/sim` directory. Use Linux x86-64 with an NVIDIA GPU,
a working driver and [Conda](https://github.com/conda-forge/miniforge#install).
Normal environment/package installation is shown below; `scripts/setup_external.sh` handles
external checkouts, CUDA-extension builds, assets and machine-path configuration.

## 1. Host tools

On Ubuntu, install missing tools:

```bash
sudo apt-get update
sudo apt-get install -y git curl build-essential ninja-build unzip ffmpeg \
  libvulkan1 vulkan-tools libgl1 libegl1 libglib2.0-0
nvidia-smi
vulkaninfo --summary
```

Vulkan must detect the NVIDIA GPU. Containers need GPU access and
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`.

## 2. Simulation environment

```bash
conda create -n codegen-sim python=3.10 pip -y
conda activate codegen-sim
conda install -c nvidia/label/cuda-12.1.1 cuda-toolkit -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
bash scripts/setup_external.sh robotwin
bash scripts/setup_external.sh configure
source env.sh
codegen_sim_preflight
"$ROBOTWIN_PY" tests/test_import_closure.py
"$ROBOTWIN_PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)'
```

If CUDA 12.1 is already installed, use its `CUDA_HOME` instead of the Conda toolkit.
The script pins RoboTwin/CuRobo, builds extensions and downloads objects, textures and robot assets.
It reuses matching unmodified Git checkouts and completed assets; it does not reset existing source.
An existing RoboTwin directory without its own Git metadata can still be used by passing
`--robotwin-root /absolute/path/to/RoboTwin` to `configure` after preparing its dependencies separately.
Do not also run RoboTwin's installer: this recipe leaves the SAPIEN/MPLib packages unpatched.

For replay alone, this environment is sufficient. A missing-slicer preflight warning must be resolved
with section 3 before marked-program validation or complete dataset production.

## 3. Dataset export and slicing

The trainer is the repository this module lives in (two levels above `codegen/sim`). It provides
`robotwin/dataprep/slicing.py` and the `src/agenticwam` LeRobot writer helpers.
A generic LeRobot package does not replace these tools.

```bash
export TRAINER_ROOT="$(cd ../.. && pwd)"     # run from codegen/sim
conda create -n codegen-data python=3.10 pip -y
conda activate codegen-data
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-data.txt
python -m pip check
export TRAIN_PY="$(command -v python)"
conda activate codegen-sim
bash scripts/setup_external.sh configure --trainer-root "$TRAINER_ROOT" --train-python "$TRAIN_PY"
source env.sh
python3 "$SLICER" --help
"$TRAIN_PY" "$SLICER" --help
"$TRAIN_PY" - <<'PY'
from capsim.data.hdf5 import downstream_root
from capsim.data.lerobot_export import check_template
downstream_root()
from agenticwam.datasets.lerobot.lerobot.datasets.compute_stats import compute_episode_stats
from agenticwam.datasets.lerobot.lerobot.datasets.utils import write_episode, serialize_dict
check_template()
print('export imports: OK')
PY
```

The compatible trainer URL/revision must be provided with the project release. The script generates
`site.yaml` and preserves unrelated existing settings; paths can also be edited using
[site.example.yaml](site.example.yaml). Stage 2 uses the active `python3` for slicing; collection
normally uses `TRAIN_PY`. `env.sh` exports `CAPX_SIM_ROOT` for placeholder slicing.
Training and policy evaluation use the trainer's own installation, beyond this data-only environment.

## 4. Claude Code and the first replay

Install [Claude Code](https://code.claude.com/docs/en/setup) if needed, then sign in:

```bash
curl -fsSL https://claude.ai/install.sh -o /tmp/claude-install.sh
bash /tmp/claude-install.sh
export PATH="$HOME/.local/bin:$PATH"
claude
```

Follow the [agent runbooks](../../.claude/README.md) with your tasks and assigned GPUs. Baseline generation
separately needs a model endpoint; see the [baseline commands](../../docs/ROBOTWIN.md).
Before long jobs, check rendering and motion on a debug seed:

```bash
mkdir -p outputs/setup-smoke
cat > outputs/setup-smoke/program.py <<'PY'
print('observation:', type(get_observation()).__name__)
print('home motion:', goto_home_joint_position('left'))
PY
CUDA_VISIBLE_DEVICES=<assigned-gpu> "$ROBOTWIN_PY" iterate/scripts/replay_trial.py \
  --code outputs/setup-smoke/program.py --task stack_blocks_two --seed 101 \
  --config iterate/configs/priv_clean.yaml --output-dir outputs/setup-smoke/replay
```

Inspect stdout and the video under `outputs/setup-smoke/oracle/replay/trial_*` for normal rendering
and no exceptions. Zero task reward is expected for this setup-only program.

## 5. Optional perception services

Skip this section for `priv_clean.yaml` and `priv_randomized.yaml`. Perception-privileged replay needs
GraspNet; full perception replay also needs SAM3 and Molmo. Their CUDA 12.8 Torch wheels require a
compatible NVIDIA driver. Create a shared SAM3/GraspNet environment:

```bash
conda create -n codegen-perception python=3.12 pip -y
conda activate codegen-perception
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-perception.txt
bash scripts/setup_external.sh perception
```

GraspNet weights are included in its checkout. Obtain [facebook/sam3](https://huggingface.co/facebook/sam3)
access, then download SAM3 weights in this environment:

```bash
hf auth login
hf download facebook/sam3 config.json sam3.pt
```

Install Molmo's serving environment separately:

```bash
conda create -n codegen-molmo python=3.12 pip -y
conda activate codegen-molmo
python -m pip install vllm==0.15.0 PyYAML
```

Start services in separate terminals from `codegen/sim`, using independently assigned GPUs:

```bash
# GraspNet terminal
conda activate codegen-perception
source env.sh
PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=<grasp-gpu> python \
  capsim/serving/launch_contact_graspnet_server.py --device cuda:0 --port 8115

# SAM3 terminal
conda activate codegen-perception
source env.sh
CUDA_VISIBLE_DEVICES=<sam3-gpu> python capsim/serving/launch_sam3_server.py --device cuda:0 --port 8114

# Molmo terminal (first start downloads weights)
conda activate codegen-molmo
CUDA_VISIBLE_DEVICES=<molmo-gpu> vllm serve allenai/Molmo2-8B \
  --host 127.0.0.1 --port 8122 --served-model-name allenai/Molmo2-8B
```

These local URLs are the client defaults; set `vision.*` in `site.yaml` for remote servers.
Check successful weight loading and exercise grounding/grasp sampling with a debug replay before
using the perception pipeline. An open HTTP port alone is not a model-readiness check.

## Later sessions

```bash
conda activate codegen-sim
cd /path/to/codegen/sim
source env.sh
codegen_sim_preflight
```

The module can be moved into a larger repository without a parent-level setup file. Update local
paths in `site.yaml` after moving the checkout. Follow the
[user guide](../../docs/ROBOTWIN.md) for validation and collection.
The recipe has been checked statically and with isolated configuration tests, not installed end-to-end
on a new GPU host.
