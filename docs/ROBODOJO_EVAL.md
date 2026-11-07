# Evaluating AgenticWAM in the RoboDojo simulator

This guide goes from a fresh clone of this repository to closed-loop scores for a RoboDojo-trained AgenticWAM
checkpoint. The checkpoint is evaluated on the dual ARX X5 in Isaac Sim. The worked example is `stack_bowls`, using the
released checkpoints `xinyuan-eval/robodojo-stack-bowls` `step{5000,10000,15000,20000}`.

**Claude Code skill:** [`.claude/skills/robodojo-pipeline`](../.claude/skills/robodojo-pipeline/SKILL.md) runs this
guide (its §6) and the steps before it: labelling, checks and training for any RoboDojo task. Start Claude Code in
the repo and ask, e.g., "evaluate step20000 on stack_bowls_random with RTC", or call `/robodojo-pipeline`. Design
decisions and every result so far are recorded in `CHOICES.md`, in the "RoboDojo sim" sections.

## How the evaluation is put together

```
 RoboDojo eval client (Isaac Sim, RoboDojo conda env)          AgenticWAM policy server (agentic-wam env)
 ───────────────────────────────────────────────────            ──────────────────────────────────────────
 layout from Assets/Eval_Layout/RoboDojo/arx_x5/        ws      deploy/robodojo/AgenticWAM/model.py
 3 cameras 480×640 + 14-D joints  ─────────────────────────▶    crop rows 16:464 → 224×320 (= training frames)
                                                                FK (X5A.urdf) → 64-D unified EEF state
                                                                model: code + 32-row action chunk
 executes 24 joint targets, 25 Hz  ◀────────────────────────    IK (deploy/common/x5_eef64.py) → joint targets
 every 24 steps → new observation                               (34 policy calls per 800-step episode)
 scores the episode (task/RoboDojo/tasks/<task>.py::get_score)
```

- **Launcher:** `eval.sh` (XPolicyLab convention, 10 positional arguments) starts the server, waits for it, then runs
  the client.
- **Episode count:** each task's count is `eval_nums` in `RoboDojo/task/RoboDojo/config/_task.yml`. For example,
  `stack_bowls_random` runs 25 episodes.
- **Score:** a per-episode score in [0, 1] plus success. For `stack_bowls`, 0.15 means one bowl stacked with both
  grippers open, and 1.0 means all three stacked. Success also needs the arms back at their start pose.
- **What the model sees:** the instruction is fixed to the training task string. Each call conditions only on the
  current frames and state; the model has no memory of earlier calls.

## 1. Requirements

- Linux x86_64 with an NVIDIA GPU of at least 48 GB. The policy (model plus the Cosmos-Reason1-7B text encoder) and
  the sim share one GPU at about 41 GB.
- The agentic-wam Python env (`docs/INSTALL.md` §2). This guide assumes `~/venvs/agenticwam-data`.
- A RoboDojo checkout with its own conda env (called `RoboDojo` here), Isaac Sim, and its assets. Follow RoboDojo's
  [install guide](https://robodojo-benchmark.com/doc/usage/install-and-download/). The training data download is
  not needed for evaluation.

## 2. One-time setup

**a. Weights and a paths file.**

1. Copy `configs/paths/default.yaml` to `configs/paths/<name>.yaml` (`docs/INSTALL.md` §3) and set:
   - `wan_vae` (Wan2.2_VAE.pth)
   - `text_encoder` (Cosmos-Reason1-7B)
   - `code_model` (the SDAR-1.7B-Chat config, tokenizer and code)
   - `code_vision` (Qwen3-VL-2B-Instruct)
2. If you use transformers ≥ 4.57, also fix SDAR's `modeling_sdar.py`: it imports `LossKwargs`, which was renamed.
   The patch is in `CHOICES.md`, "SDAR model-code fix".
3. Export `AGENTICWAM_PATHS=<name>`. Without it the server looks for `paths=r6000`, which exists only on the
   original machine.

**b. RoboDojo checkout.** Clone it inside this repo as `RoboDojo/` (git-ignored), or anywhere else and pass its path
to `install.sh`. Then, from `RoboDojo/` in its conda env:

```bash
git submodule update --init --recursive
git -C XPolicyLab checkout HEAD -- utils scripts setup_policy_server.py pyproject.toml   # if XPolicyLab files are missing
python utils/update_embodiment_config_path.py   # writes Assets/Robots/x5/curobo.yml; without it the planner crashes
```

**c. Hook the adapter in.**

```bash
cd ~/agentic-wam
bash deploy/robodojo/install.sh [path/to/RoboDojo] [path/to/agentic-wam python]
```

`install.sh` does three things:
- Symlinks `deploy/robodojo/AgenticWAM` to `RoboDojo/XPolicyLab/policy/AgenticWAM`.
- Installs the websocket server dependencies into the agentic-wam env.
- Applies `deploy/robodojo/patches/robodojo_video_every_step.patch` to RoboDojo's `src/eval_client/eval_env.py`.
  The adapter turns this on with `ROBODOJO_VIDEO_EVERY_STEP=1`, so eval videos get one frame per control step.
  Without the patch, the video has one frame per policy call (35 frames, so the arm looks about 24× too fast).

**d. Checkpoints.**

```bash
hf download xinyuan-eval/robodojo-stack-bowls --local-dir checkpoints/robodojo-stack-bowls   # step*/model.pt, ~10 GB each
```

**e. Normalisation stats.** The server reads `dataset_stats_eef64.json` from the training dataset dir. For
stack_bowls a copy ships in `deploy/robodojo/stats/stack_bowls/` and is used automatically when
`data_local/robodojo_x5_stack_bowls` is absent. For your own training run, set `AGENTICWAM_DATASET=<training view>`.

**f. Omniverse EULA.** Isaac Sim asks you to accept it on the first start. A non-interactive run reads EOF and exits
with "Unable to bootstrap inner kit kernel". Accept it once interactively, or set `OMNI_KIT_ACCEPT_EULA=YES` if you
accept the license.

## 3. Run an evaluation

```bash
cd ~/agentic-wam/RoboDojo/XPolicyLab/policy/AgenticWAM
# eval.sh <bench> <task> <ckpt> <env_cfg> <action_type> <seed> <policy_gpu> <sim_gpu> <policy_env> <sim_conda_env>
EVAL_NUM=1 bash eval.sh RoboDojo stack_bowls step20000 arx_x5 joint 0 0 0 ~/venvs/agenticwam-data RoboDojo   # smoke

# the full suite for one task, with logs for later analysis
R=step20000-$(date +%F_%H-%M-%S); L=~/agentic-wam/runs/robodojo_stack_bowls/eval/$R; mkdir -p $L
OMNI_KIT_ACCEPT_EULA=YES ROBODOJO_RUN_ID=$R \
AGENTICWAM_CODE_LOG=$L/code.jsonl AGENTICWAM_STATE_LOG=$L/state.jsonl \
  bash eval.sh RoboDojo stack_bowls step20000 arx_x5 joint 0 0 0 ~/venvs/agenticwam-data RoboDojo > $L/eval.log 2>&1
```

| Argument / variable | Meaning |
|---|---|
| `<task>` | sim task name, e.g. `stack_bowls`, `stack_bowls_random` (`RoboDojo/task/RoboDojo/tasks/*.py`) |
| `<ckpt>` | a name under `checkpoints/robodojo-stack-bowls` (`<name>/model.pt`), an absolute `.pt`, or a training run dir (newest `step_*.pt`); `AGENTICWAM_CKPT_ROOT` changes the root |
| `<action_type>` | must be `joint`: the server solves IK itself |
| `<policy_gpu>` | CUDA index or `GPU-<uuid>` |
| `<sim_gpu>` | **integer** CUDA index only. On machines with a hidden GPU, the nvidia-smi index ≠ the CUDA index |
| `EVAL_NUM` | caps the episode count (unset = the task's `eval_nums`) |
| `ROBODOJO_RUN_ID` | run name; reusing it resumes that run, a new one starts fresh |
| `AGENTICWAM_RTC=1` | real-time chunking: the previous chunk's unexecuted rows guide the next chunk |
| `AGENTICWAM_CODE_PIN=1` | keep the first program for the whole episode (default: regenerate every replan) |
| `AGENTICWAM_CODE_LOG`, `AGENTICWAM_STATE_LOG` | JSONL per policy call: the generated program; the state seen + 24 executed targets |
| `AGENTICWAM_INSTRUCTION`, `AGENTICWAM_INFER_CODE_TOKENS` | for a checkpoint trained on another task (default: the stack_bowls string, 128) |
| `AGENTICWAM_DATASET`, `AGENTICWAM_PATHS`, `AGENTICWAM_NIS`, `AGENTICWAM_REPLAN` | stats dir, paths file, denoise steps (6), rows executed per chunk (24) |

Timing: the first call compiles (about 2 minutes). After that, a 25-episode suite takes 70–90 minutes on an RTX 6000
Ada.

## 4. Results and analysis

- **Results:** `RoboDojo/eval_result/RoboDojo/<task>/AgenticWAM/arx_x5/0_ckpt_name=<ckpt>,action_type=joint/<run id>/`.
  - `_result.json` holds `score`, `success_rate`, `eval_time` (the episode count), and `details` with per-episode
    score and success.
  - `score` is the mean per-episode score × 100: four episodes at 0.15 out of 25 give 2.4, i.e. a mean of 0.024.
  - `episode_NNNNNNN_cam_{head,left_wrist,right_wrist}_{success,fail}.mp4` are the camera videos.
- **Analysis tools:** in `.claude/skills/robodojo-pipeline/scripts/`. Set `RD_VIEW=<training view>` for the scripts
  that read training data.
  - `rt_overlays.py <code.jsonl> <result dir> <out dir> <label>` burns each replan's program and a replan counter
    onto the real-time video.
  - `state_compare.py <state.jsonl> <out dir>` compares the eval states with the training states at each policy call.
  - `openloop.py` replays training episodes through the model. Nothing moves.
  - `fk_ik_check.py` and `pipeline_compare.py` check the action labels and the image pipeline against inference.

Reference numbers for `stack_bowls` checkpoints (0/25 successes in every 25-episode run):

| Setting | Episodes | Mean score |
|---|---|---|
| step20000, fixed layouts, program regenerated | 25 | 0.090 |
| step20000, fixed layouts, program pinned | 25 | 0.078 |
| step20000, fixed layouts 0–4: no RTC / RTC | 5 | 0.090 / 0.120 |
| step10000, fixed layouts 0–4 | 5 | 0.000 |
| step20000, `stack_bowls_random`: no RTC / RTC | 25 | 0.024 / 0.012 |

The typical failure: the robot stacks one bowl, then keeps re-grasping it and putting it back (`CHOICES.md`, "RoboDojo
sim evaluation: diagnosis").

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| client exits "Unable to bootstrap inner kit kernel" | EULA prompt hit EOF: §2f |
| crash building the cuRobo planner, `curobo.yml` missing | run `python utils/update_embodiment_config_path.py` in `RoboDojo/` |
| `install.sh`: `setup_policy_server.py ... is missing` | incomplete XPolicyLab submodule: §2b |
| server: `no checkpoint for ckpt_name=...` | download (§2d) or pass an absolute path |
| server: missing `configs/paths/r6000.yaml` / a weight path | create a paths file and export `AGENTICWAM_PATHS` (§2a) |
| `ImportError: LossKwargs` | SDAR file vs transformers ≥ 4.57: §2a |
| sim fails to start on the requested GPU | `<sim_gpu>` must be an integer CUDA index |
| CUDA OOM | another job on the GPU; the eval needs ~41 GB free |
| videos look sped up | the eval_env patch is not applied (§2c) |
