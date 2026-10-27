# RoboTwin

Evaluate the released WAM and hybrid models in RoboTwin 2.0. WAM predicts action chunks; hybrid executes accepted generated code and uses the WAM action head as fallback. Both use three cameras and 14-D absolute joint actions.

## Setup

Install the model environment and public model components in [INSTALL.md](INSTALL.md). Prepare RoboTwin, cuRobo, Vulkan, and the robot/object assets using [codegen/sim/SETUP.md](../codegen/sim/SETUP.md). Hybrid also needs the Molmo, SAM3, and Contact-GraspNet environments described there.

From the repository root:

```bash
hf download miniFranka/agentic-wam-posttrain-robotwin-suit --local-dir checkpoints/robotwin_release
cp configs/robotwin/site.example.yaml configs/robotwin/site.yaml
```

Set your Python, simulator, library, and Hugging Face cache paths in `site.yaml`. Set the public component paths (`wan_vae`, `text_encoder`, `code_model`, `code_vision`) in `configs/paths/default.yaml`. Download those components before evaluation; the launcher reads the policy cache offline.

The release contains:

| Model | Weights | Execution |
|---|---|---|
| WAM | `wam.pt` | WAM action head |
| Hybrid | `hybrid.pt` | Generated code with WAM fallback |

`dataset_stats.json` is shared by both models. Keep `task_skill_prompt.json` beside the weights. The hybrid task programs are included in `configs/robotwin/tasks_indep/`.

## Evaluation

Start with one episode in each clean/randomized setting:

```bash
python scripts/eval_robotwin_release.py wam --tasks click_alarmclock --plan smoke --gpus 0
python scripts/eval_robotwin_release.py hybrid --tasks click_alarmclock --plan smoke --gpus 0 --vis-gpu 1
```

The two test-time configs are [wam.yaml](../configs/robotwin/eval/wam.yaml) and [hybrid.yaml](../configs/robotwin/eval/hybrid.yaml). The commands select the matching config and weights automatically.

`--vis-gpu` starts or checks the hybrid perception services. `--vis-gpu-seg` can place SAM3 and GraspNet on another GPU. Without `--vis-gpu`, the services must already be running. Reserve perception GPUs separately from policy GPUs.

For the complete 50-task benchmark, with 100 episodes per task and setting:

```bash
python scripts/eval_robotwin_release.py wam --gpus 0
python scripts/eval_robotwin_release.py hybrid --gpus 0 --vis-gpu 1
```

Useful options:

| Option | Purpose |
|---|---|
| `--tasks a,b` | Evaluate selected tasks |
| `--protocols clean` or `randomized` | Select one scene setting; default is both |
| `--gpus 0,1` | Schedule policy jobs across these GPUs |
| `--episodes N --shards S` | Set the episode count and number of jobs per task/setting |
| `--name my_eval` | Choose an output name |
| `--ckpt /path/to/wam.pt` | Evaluate another weights file |
| `--render` | Save rollout videos |
| `--dry-run` | Print the plan without starting services or simulation |

Instructions default to the `seen` pool. To use the `unseen` pool, append `--launch-args --instruction-type unseen` and choose another output name.

For a one-GPU H100 80 GB smoke test, set `vision.molmo.gpu_util: 0.23` in `site.yaml` and run:

```bash
AGENTICWAM_TEXT_OFFLOAD=1 python scripts/eval_robotwin_release.py hybrid \
  --tasks click_alarmclock --plan smoke --gpus 0 --vis-gpu 0
```

This caches GPU-computed instruction embeddings and moves the text encoder to CPU between instructions. Use separate policy and perception GPUs for multi-episode benchmarks.

## Results

Benchmark reports are written to `evaluate_results/robotwin_benchmark/<name>/`; per-episode results and effective settings are under `evaluate_results/robotwin/`. A completed job is reused only when its process succeeded and its episode count matches the request. Rerun the same command to resume.

```bash
python robotwin/eval/report.py <name>
```

To browse rendered rollouts:

```bash
python robotwin/hybrid_eval/rollout_viewer.py build
python robotwin/hybrid_eval/rollout_viewer.py serve
```

## Data collection

`codegen/sim` executes validated task programs and records the code responsible for each motion. Its exporter writes LeRobot data, code-stage slices, and a `dataset.yaml` manifest. Follow [codegen/sim/SETUP.md](../codegen/sim/SETUP.md) and the collection commands in `scripts/posttrain_annotation_robotwin.sh`.

Programs use `[setup]` for scene preparation and `[stage k/N]` for motion stages. Keep scene-specific geometry in perception calls; the frozen program and recorded execution provide the code labels. The collector calls `robotwin/dataprep/slicing.py` to build the aligned stage cache.

Prepare collected manifests for training:

```bash
python robotwin/dataprep/prepare.py --datasets /path/to/dataset.yaml --name my_tasks
```

This builds the instruction-embedding cache and writes `configs/robotwin/data/my_tasks.yaml` and `configs/robotwin/task/my_tasks.yaml`. Machine paths come from the `dataprep` section of `site.yaml`. `--skip-text-cache` reuses an existing cache; `--split all` includes all kept episodes.

## Training

Configure the dataset and model paths, then train with the appropriate task config:

```bash
bash scripts/train_robotwin.sh robotwin_wam
bash scripts/train_robotwin.sh robotwin_hybrid
# Or use the config generated for your data:
bash scripts/train_robotwin.sh my_tasks
```

`robotwin_wam` uses the official RoboTwin demonstrations; `robotwin_hybrid` uses code-labeled data collected with `codegen/sim`. The launcher runs preflight checks before training. Override training duration, batch size, learning rate, and save/evaluation intervals through the task config or Hydra arguments.

Training outputs go to `runs/<task>/<run_id>/`. Set `RUN_ID` to resume a run. `DRY_RUN=1` prints the training command after preflight. Multi-node training uses `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`.
