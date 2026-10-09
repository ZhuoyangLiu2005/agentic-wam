---
name: downstream-training
description: "Launch, monitor and resume AgenticWAM downstream training: real-robot post-training (YAM, Tianji + Sharpa) from the AW-0 checkpoint and RoboTwin training (robotwin_wam, robotwin_hybrid or a config from robotwin-dataprep), single- or multi-node, with the preflight dry run as the gate; then point to serving or the RoboTwin benchmark."
---

# Launch downstream training

Guides: real robots [docs/REALWORLD.md](../../../docs/REALWORLD.md) §3, RoboTwin
[docs/ROBOTWIN.md](../../../docs/ROBOTWIN.md) §4, multi-node and launch internals
[docs/PRETRAINING.md](../../../docs/PRETRAINING.md) §4. Entry points:

| target | command |
|---|---|
| YAM | `bash scripts/train_real_yam.sh <dataset> <code_cache.pt> [overrides]` |
| Tianji + Sharpa | `bash scripts/train_real_sharpa.sh <dataset> <code_cache.pt> [overrides]` |
| RoboTwin | `bash scripts/train_robotwin.sh <robotwin_wam \| robotwin_hybrid \| <name>> [overrides]` |

Each runs `scripts/preflight_sft.py` (config composes, robot transform, every path, code-cache coverage, instruction
embeddings, idle GPUs) and then `scripts/launch_train.py` (accelerate + DeepSpeed ZeRO-1).

## Ask first

1. **Target and data**: robot + dataset dir + code cache (from `posttrain-annotation`), or the RoboTwin task config
   (from `robotwin-dataprep`).
2. **Compute**: GPUs per node (`NPROC`, default 8), number of nodes, and whether the GPUs are free. Never launch onto
   GPUs another job is using.
3. **Run name** (`RUN_ID`): needed to resume and for multi-node; default is a timestamp.
4. **Logging**: `WANDB_API_KEY` exported by the user, or `wandb.enabled=false` / `WANDB_MODE=offline`. Never write
   the key anywhere.
5. **Recipe changes**, if any (`max_steps`, `batch_size`, `save_every`, `model.infer_code_tokens`, …). The defaults are
   the paper recipes; change them only when asked.

## Steps

**1. Environment.** The `agenticwam` env with `PYTHONPATH=<repo>/src` first, `HF_HOME`, and
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` on offline nodes; `PYTHON=<env python>` if it is not `python` on `PATH`. A
persistent `HF_DATASETS_CACHE` avoids rebuilding the LeRobot arrow cache on every launch. If a per-machine paths file is
in use, append `paths=<name>` to every command.

**2. Gate: dry run.** It must pass before anything starts:

```bash
DRY_RUN=1 bash scripts/train_real_yam.sh <dataset> <code_cache.pt> [overrides]
```

Fix every reported problem (missing path, uncovered frames, unembedded instruction, busy GPU) instead of skipping the
preflight. `SKIP_PREFLIGHT=1` exists but is for debugging only.

**3. Smoke** (recommended on a new machine or dataset): a few real steps prove the weights load and the loss is finite.

```bash
bash scripts/train_real_yam.sh <dataset> <code_cache.pt> max_steps=3 save_every=3 wandb.enabled=false output_dir=/tmp/smoke
```

**4. Launch.** Confirm the final command with the user, then start it detached and logged (a training run takes days):

```bash
RUN_ID=<run> nohup bash scripts/train_real_yam.sh <dataset> <code_cache.pt> > train_<run>.log 2>&1 &
# multi-node: the same command on every node, plus NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT and the SAME RUN_ID
```

Two runs on one machine need different `MASTER_PORT` values and disjoint `CUDA_VISIBLE_DEVICES`.

**5. Monitor.** The log shows the foundation transfer (`transferred N/N tensors`), the trainable modules, then step
lines; W&B has `train/loss`, its per-stream components (`train/*`), `train/grad_norm` and `train/lr`. Report early: a NaN or exploding loss, a
step rate far below the first minutes, or a stalled log. Checkpoints land in
`./runs/<task>[_<dataset>]/<RUN_ID>/checkpoints/weights/step_XXXXXX.pt`.

**6. Resume** with the same `RUN_ID` and `resume=<checkpoint>`. Recipes save weights only by default
(`save_optimizer_state: false`), so a resume restarts the optimizer state.

## Recipe facts worth knowing

* Real-robot recipe: 100k steps, batch 4 per GPU × 8 GPUs, lr 1e-4 cosine, `CFG_DROPOUT=0.15,0.0`, starts from
  `paths.pretrain_ckpt`. The data configs hold out no episodes, so the periodic eval runs on training episodes.
* `model.infer_code_tokens` (default 64) must cover the longest code label and must be the same value when serving.
* `gradient_accumulation_steps` changes the effective batch, not speed. `ZERO=2` switches to ZeRO-2 if memory is tight.
* `model.use_code_expert=false` trains the video + action baseline without the code expert.

## After training

* **YAM**: serve with `deploy/launch_server.sh` (REALWORLD.md §4; `INFER_CODE_TOKENS` = the training value).
* **RoboTwin**: `python robotwin/eval/benchmark.py --model <preset> --ckpt <step.pt>` (ROBOTWIN.md §5; start with
  `--plan smoke`). The eval links `robotwin/deploy` into the RoboTwin checkout as `EVALUATION.policy_name`; if another
  checkout already owns that link, give this one its own `EVALUATION.policy_name`.
