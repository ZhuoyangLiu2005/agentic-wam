---
name: agenticwam-setup
description: "Set up AgenticWAM on a new machine: the conda environment, the public model weights and the AW-0 checkpoint, the paths file, environment variables, and the smoke tests that prove the install works. Also points to the separate RoboTwin and codegen/sim setups when the user needs simulation."
---

# Set up the environment

The user-facing guide is [docs/INSTALL.md](../../../docs/INSTALL.md); this skill is the order to do it in and the checks
between steps. Run every command from the repository root.

## Ask first

1. **What will this machine do?** Real-robot post-training / serving, RoboTwin training + evaluation, RoboTwin data
   collection (`codegen/sim`), pretraining, or code labeling. Only install what the answer needs.
2. **Where should weights live?** A directory with ~40 GB free (the AW-0 checkpoint alone is ~10 GB).
3. **Does the node have internet?** If not, weights must be downloaded elsewhere and copied in, and the offline
   variables in step 4 are mandatory.
4. **A per-machine paths file or edit the default?** Shared checkouts should use `configs/paths/<machine>.yaml` plus
   `paths=<machine>` on every command, so `default.yaml` stays a template.

## Steps

**1. Environment** (Python 3.10, CUDA 12.8):

```bash
conda create -n agenticwam python=3.10 -y && conda activate agenticwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Check: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` prints `2.7.1+cu128 True`, and
`python -c "import transformers; print(transformers.__version__)"` is `>= 4.57` and `< 5`.

**2. Weights.** Download only what the job needs (table in INSTALL.md §3):

| job | keys it needs |
|---|---|
| real-robot post-training | `wan_vae`, `code_model`, `code_vision`, `pretrain_ckpt` (+ `text_encoder` to embed new instructions) |
| serving on YAM | the same, plus `text_encoder` |
| RoboTwin training | `wan_vae`, `code_model`, `code_vision`, `text_encoder`, `robotwin_*_init_ckpt`, `robotwin_official*` or `robotwin_cap*` |
| pretraining from scratch | the above plus `cosmos_ckpt` (not distributed — ask the user where theirs is) |

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir <weights>/Wan2.2-TI2V-5B
huggingface-cli download nvidia/Cosmos-Reason1-7B --local-dir <weights>/Cosmos-Reason1-7B
huggingface-cli download JetLM/SDAR-1.7B-Chat --local-dir <weights>/SDAR-1.7B-Chat
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir <weights>/Qwen3-VL-2B-Instruct
huggingface-cli download miniFranka/agentic-wam-pretrain-yamego3k-soup-e1 --local-dir <weights>/aw0-pretrain
```

**3. Paths.** Fill the keys from step 2 in `configs/paths/default.yaml` (or a copy, see "Ask first"). Every value must
be an absolute path that exists; leave unused keys as their `/path/to/...` placeholder. Verify:

```bash
python - <<'PY'
import os, yaml
p = yaml.safe_load(open("configs/paths/default.yaml"))        # or configs/paths/<machine>.yaml
for k, v in p.items():
    print(f"{'ok ' if os.path.exists(str(v)) else '-- '} {k}: {v}")
PY
```

**4. Environment variables** (put them in the user's shell profile or job script, never in a tracked file):

```bash
export PYTHONPATH=<repo>/src:${PYTHONPATH:-}
export HF_HOME=<hf_cache>
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1      # offline nodes, once weights are local
export WANDB_API_KEY=<key>                         # or train with wandb.enabled=false
```

Never write `WANDB_API_KEY`, a Hugging Face token or a VLM key into a file in the repository, and never echo them.

**5. Smoke tests**, in order; stop at the first failure and use the troubleshooting table (INSTALL.md §6):

```bash
python -c "import agenticwam, inspect; print(inspect.getfile(agenticwam))"    # must be <repo>/src/agenticwam/__init__.py
python scripts/launch_train.py --task robotwin_wam --dry-run                   # configs compose
DRY_RUN=1 bash scripts/train_real_yam.sh <dataset_dir> <code_cache.pt>        # if the user already has a dataset
```

With a GPU free and a dataset at hand, a few real steps load every weight:
`bash scripts/train_real_yam.sh <dataset_dir> <code_cache.pt> max_steps=3 save_every=3 wandb.enabled=false
output_dir=/tmp/agenticwam_smoke`. The log must show the foundation transfer and a finite loss.

## Simulation and labeling environments (only if asked)

* **RoboTwin evaluation**: a RoboTwin 2.0 checkout with Vulkan + cuRobo, and `configs/robotwin/site.yaml` copied from
  `site.example.yaml` — [docs/ROBOTWIN.md](../../../docs/ROBOTWIN.md) `vulkaninfo --summary` must list the NVIDIA GPU.
* **RoboTwin data collection** (`codegen/sim`): its own conda environments — follow `codegen/sim/SETUP.md` and finish with
  `source env.sh && codegen_sim_preflight` from `codegen/sim`.
* **Code labeling**: `pip install -r codegen/pretrain/pipeline/requirements.txt` in the env that will run it, and an
  OpenAI-compatible VLM endpoint (`vlm_url` in the engine config, key in `VLM_API_KEY`).

## Done when

The smoke tests in step 5 pass, `configs/paths/*.yaml` names only existing files for the keys this machine uses, and
the user knows which `paths=<name>` (if any) to pass. Tell them the next skill: `posttrain-annotation` /
`downstream-training` for a real robot, `robotwin-dataprep` for RoboTwin, `pretrain-annotation` for pretraining data.
