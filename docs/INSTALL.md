# Installation

AgenticWAM is a tri-modal **video · code · action** World-Action Model: a Cosmos-Predict2.5-2B video expert with the
Wan2.2 VAE, a code expert (SDAR-1.7B body, code co-denoised with video and action) and a diffusion action head,
coupled through a Mixture-of-Transformers with per-layer joint self-attention. This guide sets up the environment,
the model weights and the paths every config reads.

## 1. Requirements

| | |
|---|---|
| GPU | H100-class; training scripts default to 8 GPUs per node |
| CUDA | 12.8 |
| Python | 3.10 |
| PyTorch | `torch==2.7.1+cu128`, `torchvision==0.22.1+cu128` |
| transformers | `>=4.57,<5` (Qwen3-VL) |

The other pins (`accelerate`, `deepspeed`, `hydra-core`, `wandb`, …) are in `pyproject.toml`.

## 2. Environment

```bash
conda create -n agenticwam python=3.10 -y && conda activate agenticwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .                     # from the repository root
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}     # also what the training scripts set
```

RoboTwin simulation and the `codegen/sim` perception servers need their own setup: [ROBOTWIN.md](ROBOTWIN.md) §1.
The real-robot model server needs only this environment: [REALWORLD.md](REALWORLD.md) §4.

## 3. Weights and paths

Every machine-specific location lives in one file, `configs/paths/default.yaml`; all configs refer to its keys as
`${paths.<key>}`. Nothing else in the repository names a path on your disk.

| key | what | where to get it |
|---|---|---|
| `wan_vae` | the frozen video VAE: the file `Wan2.2_VAE.pth` | Hugging Face `Wan-AI/Wan2.2-TI2V-5B` |
| `text_encoder` | the instruction encoder (instruction embeddings are precomputed; also used when serving) | Hugging Face `nvidia/Cosmos-Reason1-7B` |
| `code_model` | the code expert's language body | Hugging Face `JetLM/SDAR-1.7B-Chat` |
| `code_vision` | the code expert's frozen vision tower | Hugging Face `Qwen/Qwen3-VL-2B-Instruct` |
| `pretrain_ckpt` | the AW-0 pretrained checkpoint: the start of real-robot post-training | the release ([README](../README.md#model-weights)) or your own pretraining run |
| `cosmos_ckpt` | pretraining **from scratch** only: a Cosmos-Predict2.5-2B video checkpoint (`model-*-of-*.safetensors`) the video expert starts from | not distributed ([PRETRAINING.md](PRETRAINING.md) §3) |
| `robotwin_wam_init_ckpt`, `robotwin_hybrid_init_ckpt` | the weights RoboTwin training starts from | a pretraining checkpoint or a released one ([ROBOTWIN.md](ROBOTWIN.md) §4) |
| `robotwin_root` | a RoboTwin 2.0 checkout (simulator, assets, eval scripts) | [ROBOTWIN.md](ROBOTWIN.md) §1 |
| `robotwin_official`, `robotwin_official_labels` | the official RoboTwin 2.0 demonstrations and their code labels | [ROBOTWIN.md](ROBOTWIN.md) §3.3 |
| `robotwin_cap`, `robotwin_cap_slices` | data collected with `codegen/sim`, and its stage slices | [ROBOTWIN.md](ROBOTWIN.md) §2–3 |

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir /data/weights/Wan2.2-TI2V-5B
huggingface-cli download nvidia/Cosmos-Reason1-7B --local-dir /data/weights/Cosmos-Reason1-7B
huggingface-cli download JetLM/SDAR-1.7B-Chat --local-dir /data/weights/SDAR-1.7B-Chat
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir /data/weights/Qwen3-VL-2B-Instruct
```

Set the keys one of three ways:

* **Edit `configs/paths/default.yaml`** — the simplest for a single machine.
* **A file per machine:** copy it to `configs/paths/<name>.yaml`, edit, and add `paths=<name>` to every command
  (e.g. `bash scripts/train_real_yam.sh <dataset> <cache.pt> paths=cluster`).
* **Single keys on the command line:** `paths.pretrain_ckpt=/data/ckpt/agenticwam_pretrain.pt`.

Only the keys a run uses need real values: real-robot post-training reads `wan_vae`, `code_model`, `code_vision`
and `pretrain_ckpt`; a `DRY_RUN=1` of the train scripts reports any path that does not exist. Tools outside Hydra
(`scripts/precompute_cosmos_text_embeds.py`, `robotwin/dataprep/text_cache.py`) default to `text_encoder` from the same
file. RoboTwin evaluation and data preparation read a second, machine-specific file, `configs/robotwin/site.yaml`
(copied from `site.example.yaml`; [ROBOTWIN.md](ROBOTWIN.md) §1).

## 4. Environment variables

```bash
export PYTHONPATH=<repo>/src:${PYTHONPATH:-}           # resolve `import agenticwam` to this checkout
export HF_HOME=/path/to/hf_cache                        # Hugging Face cache (text encoder, tokenizers)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1          # on nodes without internet, after the weights are local
export WANDB_API_KEY=<key>                              # from the environment only; or pass wandb.enabled=false
```

Training logs to wandb by default (`configs/train.yaml`, project `agenticwam`); never put the key in a script or
config.

**Inference precision.** Training runs bf16. The simulation policy and the real-robot runtime load the model in
fp32 unless `AGENTICWAM_DEPLOY_DTYPE=bf16` is set; the real-robot server (`deploy/launch_server.sh`) sets bf16, which
halves load time.

## 5. Smoke test

```bash
# (a) the package resolves to this checkout
python -c "import agenticwam, inspect; print(inspect.getfile(agenticwam))"     # <repo>/src/agenticwam/__init__.py
# (b) a recent enough transformers
python -c "import transformers; print(transformers.__version__)"             # >= 4.57
# (c) a task end to end without training: composes the config, checks every path, the data, code cache and GPUs,
#     and prints the launch command
DRY_RUN=1 bash scripts/train_real_yam.sh <dataset_dir> <code_cache.pt>
```

A few real steps load the weights too: `bash scripts/train_real_yam.sh <dataset_dir> <code_cache.pt> max_steps=6
save_every=3 output_dir=/tmp/smoke`. The log shows the foundation transfer (`transferred N/N tensors`), the trainable
modules and the loss.

## 6. Troubleshooting

| symptom | fix |
|---|---|
| `import agenticwam` resolves outside this checkout | put `<repo>/src` first on `PYTHONPATH` |
| unknown `model_type qwen3_vl` | `transformers>=4.57` |
| a path under `/path/to/...` in an error | a `paths.*` key is still the placeholder (§3) |
| Hugging Face tries the network on an offline node | set `HF_HOME` and `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` |
| `torch.compile` / Dynamo errors | the train scripts set `TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1`; set them when calling `scripts/launch_train.py` directly |
| RoboTwin eval crashes in Vulkan / cuRobo | the simulator needs Vulkan and cuRobo; [ROBOTWIN.md](ROBOTWIN.md) §1 and §5.2 |
| intermittent multi-GPU "illegal memory access" | usually a node left in a bad state by another job; retry on a clean node first |

## 7. Next

[DATA.md](DATA.md) (real-robot data) · [REALWORLD.md](REALWORLD.md) (post-training and deployment) ·
[PRETRAINING.md](PRETRAINING.md) · [ROBOTWIN.md](ROBOTWIN.md) · [CODE_ANNOTATION.md](CODE_ANNOTATION.md)
