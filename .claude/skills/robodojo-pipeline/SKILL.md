---
name: robodojo-pipeline
description: "End-to-end AgenticWAM post-training for any RoboDojo sim task (dual ARX X5, LeRobot v2.1 release): training view, code labelling with the VLM engine, label/kinematics/image checks, inference-vs-training input checks, post-training from AW-0, open-loop replay, and closed-loop Isaac Sim evaluation with real-time code-overlay videos. Use when the user names a RoboDojo task (stack_bowls, hang_mugs, …) or a RoboDojo dataset and wants it labelled, checked, trained or evaluated. Paid VLM stages and long GPU jobs need the user's explicit go-ahead."
---

# RoboDojo task → labelled, trained, evaluated AgenticWAM policy

Runs from `~/agentic-wam`. This was first done for stack_bowls; `CHOICES.md` ("Choices: RoboDojo sim", "RoboDojo sim
evaluation") records every decision and result from that run. Read it before changing a default. Sub-runbooks used
here: [posttrain-annotation](../posttrain-annotation/SKILL.md) and [downstream-training](../downstream-training/SKILL.md).
Check scripts live in `scripts/` next to this file and read `RD_VIEW` (plus `RD_SRC`, `RD_TASK_INDEX`,
`RD_TRAIN_EPS`; see `scripts/rd_env.py`).

Fixed facts (do not re-derive):
- **Data.** Source: `RoboDojo/data/RoboDojo_lerobot_v21_video`, 35 tasks × 100 episodes at 25 Hz. State is 14-D joints
  `[L q0..5, L grip, R q0..5, R grip]`, and `action[t] = state[t+1]`. Video is AV1 480×640.
- **Converted layout.** The YAM 82-D layout: link6 FK poses from `X5A.urdf`, grippers at [32]/[33] (1 = open), joints
  at [34:46]. Video is cropped to rows 16:464 and area-resized to 224×320. Because of this layout, the stock
  `posttrain_yam` task, `YamUnifiedEEFTransform` and `scripts/train_real_yam.sh` apply unchanged.
- **Serving.** The deploy adapter applies the same crop, runs IK through `deploy/common/x5_eef64.py`, replans every
  24 steps, uses 32-row chunks and 6 denoise steps.
- **d3 GPU numbering.** nvidia-smi index ≠ CUDA index: GPU 4 is hidden, so nvidia-smi 5/6/7 are CUDA 4/5/6.
  - Policy and training GPUs are selected by `GPU-<uuid>`.
  - The sim's `env_gpu` must be an integer CUDA index.

## Ask first

1. **Task.** Get the sim task name (`RoboDojo/task/RoboDojo/tasks/<name>.py`) and find its source `task_index` with
   `grep -n templates RoboDojo/task/RoboDojo/tasks/<name>.py`. Match that string against
   `RoboDojo/data/RoboDojo_lerobot_v21_video/meta/tasks.jsonl` and confirm both with the user.
2. **Which stages.** The full sequence, or a subset. Each stage below has a gate; never skip a gate.
3. **Labelling budget.** The VLM key comes from `VLM_API_KEY` or `codegen/pretrain/pipeline/.vlm_api_key`. Get a tag
   (`robodojo_<task>_v1`). Report the call counts from the dry run and get an explicit yes before each paid stage.
4. **GPUs.**
   - Training: ≥ 4 free 48 GB GPUs, ideally 8. One GPU OOMs at any batch size.
   - Eval: one GPU shared by the policy (~41 GB with the sim).
   - Run `nvidia-smi` and never launch onto a GPU another job is using.
5. **Eval size.** `EVAL_NUM` defaults to 5 episodes (the user's standing choice for stack_bowls).

Set once per task:

```bash
cd ~/agentic-wam
PY=~/venvs/agenticwam-data/bin/python
TASK=<sim task name>  K=<source task_index>  INSTR="<the task string>"
V=$PWD/data_local/robodojo_x5_$TASK          # training view
K_DIR=.claude/skills/robodojo-pipeline/scripts
export RD_VIEW=$V
```

## 1. Training view

```bash
# a. convert (skip if data_local/robodojo_x5_yam82/train_shard_$K exists: all 35 tasks are converted on d3)
$PY utils/convert_robodojo_v21_to_yam82.py --src RoboDojo/data/RoboDojo_lerobot_v21_video \
    --dst data_local/robodojo_x5_yam82 --tasks $K --num_workers 16
# b. view = one shard renamed to train_shard_0, its own stats, the shared instruction cache, 95/5 split
mkdir -p $V && ln -sfn ../robodojo_x5_yam82/train_shard_$K $V/train_shard_0
ln -sfn ../robodojo_x5_yam82/cosmos_text_embeds_cache $V/cosmos_text_embeds_cache
$PY utils/make_stats.py --transform yam --dst $V            # -> $V/dataset_stats_eef64.json
$PY -c "import json;json.dump(list(range(95)),open('$V/episodes_0_94.json','w'));json.dump(list(range(95,100)),open('$V/episodes_95_99.json','w'))"
$PY utils/verify_instruction_embeds.py --dataset $V
```

**Gate:**
- `verify_instruction_embeds` reports mismatch 0 and embed_miss 0. If the string is missing, run
  `scripts/precompute_cosmos_text_embeds.py --instructions <file> --out-dir $V/cosmos_text_embeds_cache --ckpt ~/models/Cosmos-Reason1-7B`.
  First replace the symlink with a real directory.
- Look at one frame per camera by eye.

## 2. Code labels (route A, `yam_sft` engine, frozen `yam` library)

```bash
export CODEGEN_CONFIG=configs/yam_sft_robodojo.yaml YAM_SFT_DATASET=$V PYTHON=$PY TAG=robodojo_${TASK}_v1
bash scripts/posttrain_annotation_yam.sh subtasks --dry-run --workers 16     # free: gripper splits
mv $V/train_shard_0/subtasks.json $V/train_shard_0/subtasks.dry.json        # else the paid run skips these episodes
# PAID, after the user's yes. Pilot first, then the rest:
bash scripts/posttrain_annotation_yam.sh subtasks --workers 8 --episodes 5
seq 0 4 | sed 's/^/train_shard_0:/' > $V/units_pilot.txt
bash scripts/posttrain_annotation_yam.sh label --tag $TAG --parallel 5 --tasks-file $V/units_pilot.txt
bash scripts/posttrain_annotation_yam.sh progress --tag $TAG
#   read the pilot programs (codegen/pretrain/pipeline/outputs/yam_sft/), show them, then repeat for all 100:
bash scripts/posttrain_annotation_yam.sh subtasks --workers 8
seq 0 99 | sed 's/^/train_shard_0:/' > $V/units.txt
bash scripts/posttrain_annotation_yam.sh label --tag $TAG --parallel 5 --tasks-file $V/units.txt --skip-done
# free: flat cache in the training view's order + alignment
bash scripts/posttrain_annotation_yam.sh cache --train-view $V --task train_shard_0 --tag $TAG \
     --work $V/annot_work --out $V/policy_code_idx_$TAG.pt
cp $V/annot_work/episode_order.json $V/episode_order.json
bash scripts/posttrain_annotation_yam.sh verify --config configs/real_post_train/data/yam.yaml \
     --dataset-root $V --order $V/episode_order.json
```

`yam_frame_gap: 33` in the config is the 25 Hz equivalent of YAM's 40 frames at 30 fps. Never edit
`codegen/pretrain/skill_library/` while labelling: doing so invalidates the stored labels.

**Gate:** `verify` prints ALIGNED with 0 missing and 0 frames without code. The progress report shows every unit
reconciled. Report matched_raw and repaired_share; stack_bowls had 0.75 / 0.22. Then derive the two numbers training
needs:

```bash
$PY - <<EOF
import json, torch
from transformers import AutoTokenizer
V="$V"; c=torch.load(f"{V}/policy_code_idx_$TAG.pt", weights_only=False)
lens=[json.loads(l)["length"] for l in open(f"{V}/train_shard_0/meta/episodes.jsonl")]
print("val frame_offset =", sum(lens[:95]), " total", sum(lens), " cache", len(c["code_id_by_idx"]))
tok=AutoTokenizer.from_pretrained("$HOME/models/SDAR-1.7B-Chat-tok", trust_remote_code=True)
L=[len(tok(s).input_ids) for s in c["codes"]]; n=sorted(L[i] for i in c["code_id_by_idx"].tolist())   # per frame
print("code tokens per frame p50/p95/p99/max", *(n[int(q*(len(n)-1))] for q in (.5,.95,.99,1)), "| frames over 128:", sum(x>128 for x in n))
EOF
```

- `OFF` = the printed val frame_offset. Without `data.val.frame_offset`, val silently reads episode 0's labels.
- Keep `INFER_TOK=128` unless the per-frame p95 is above it (stack_bowls: p95 93, 4.7% of frames over 128; truncation applies at inference only).

## 3. Label and input checks (free, before training)

```bash
$PY -I $K_DIR/fk_ik_check.py                 # labels use X5A.urdf; FK(stored joints)=stored pose; encode/decode round trip
$PY -I $K_DIR/pipeline_compare.py 0 runs/robodojo_$TASK/checks   # training vs inference image pipeline, every stage
```

**Gate:**

| Check | Pass condition | stack_bowls value |
|---|---|---|
| `fk_ik_check` FK | URDF vs `X5_CHAIN` ≈ 0 mm; stored vs FK < 0.01 mm | 0.0 mm / 3e-5 mm |
| `fk_ik_check` labels | `action[t]==state[t+1]`; encode error ~1e-7 | ✓ / 8e-8 |
| `fk_ik_check` decode | 0 IK failures; max position error ≤ ~0.1 mm | 0 / 0.1 mm |
| `pipeline_compare` | final [-1, 1] input mean\|d\| < 0.02, from the H.264 re-encode only | 0.0148 |

Look at the `pipe_*.png` images. If any check fails, stop and diagnose; do not train on bad labels.

## 4. Train (AW-0 → task), via downstream-training's gates

```bash
COMMON=(paths=r6000 model.infer_code_tokens=$INFER_TOK
        data.train.episode_indices=$V/episodes_0_94.json data.val.episode_indices=$V/episodes_95_99.json
        data.val.frame_offset=$OFF)
DRY_RUN=1 NPROC=8 PYTHON=$PY bash scripts/train_real_yam.sh $V $V/policy_code_idx_$TAG.pt "${COMMON[@]}"
NPROC=8 PYTHON=$PY bash scripts/train_real_yam.sh $V $V/policy_code_idx_$TAG.pt "${COMMON[@]}" \
    max_steps=3 save_every=3 wandb.enabled=false output_dir=/tmp/smoke_$TASK
# after the user confirms GPUs + command:
RUN_ID=robodojo_${TASK}_v1 CUDA_VISIBLE_DEVICES=<8 free GPUs, UUIDs on d3> NPROC=8 PYTHON=$PY \
nohup bash scripts/train_real_yam.sh $V $V/policy_code_idx_$TAG.pt "${COMMON[@]}" \
    max_steps=20000 save_every=5000 eval_every=1000 > runs/train_robodojo_$TASK.log 2>&1 &
```

**Dry-run gate.** Every line must be `[ok]`, except two expected warnings:
- `cache is longer` (the train split covers the first `OFF` frames)
- `gpus skipped`

**Monitoring.**
- The log shows `transferred N/N tensors`, then a finite, falling loss.
- Checkpoints land in `runs/posttrain_yam_robodojo_x5_$TASK/<RUN_ID>/checkpoints/weights/step_XXXXXX.pt`.
- Choose the checkpoint where val loss bottoms out. A 100-episode task overfits well before 20k steps.

**Resume.** `resume=<step_*.pt>` with the same `RUN_ID`. It restores weights only; the optimizer state restarts.

## 5. Open-loop replay (free, nothing moves)

```bash
CUDA_VISIBLE_DEVICES=<gpu> $PY -I $K_DIR/openloop.py --ckpt <step_*.pt> --dataset $V \
    --episodes 0 97 --infer-code-tokens $INFER_TOK --out runs/robodojo_$TASK/openloop/<step>
```

Use a trained episode and a held-out episode. Gate: on stack_bowls, EEF error was a few mm and < 1° per arm, and the
generated program followed the training subtask order. Open loop says nothing about closed-loop success, so always
continue to §6.

## 6. Closed-loop sim eval (Isaac Sim, one GPU)

User-facing guide for this stage, including setup from a fresh clone: [docs/ROBODOJO_EVAL.md](../../../docs/ROBODOJO_EVAL.md).

Once per machine:
- `bash deploy/robodojo/install.sh`
- `cd RoboDojo && python utils/update_embodiment_config_path.py` in the RoboDojo conda env. It writes
  `Assets/Robots/x5/curobo.yml`; without it the planner crashes.

```bash
cd ~/agentic-wam/RoboDojo/XPolicyLab/policy/AgenticWAM
G=<policy GPU UUID>  E=<same GPU's CUDA index>  R=$TASK-<step>-$(date +%F_%H-%M-%S)
L=~/agentic-wam/runs/robodojo_$TASK/eval/$R; mkdir -p $L
EVAL_NUM=5 OMNI_KIT_ACCEPT_EULA=YES ROBODOJO_RUN_ID=$R \
AGENTICWAM_DATASET=$V AGENTICWAM_INSTRUCTION="$INSTR" AGENTICWAM_INFER_CODE_TOKENS=$INFER_TOK \
AGENTICWAM_CODE_LOG=$L/code.jsonl AGENTICWAM_STATE_LOG=$L/state.jsonl \
  bash eval.sh RoboDojo $TASK <abs step_*.pt or run dir> arx_x5 joint 0 $G $E ~/venvs/agenticwam-data RoboDojo \
  > $L/eval.log 2>&1
```

**Options.**
- `AGENTICWAM_RTC=1` adds real-time chunking: the previous chunk's unexecuted rows guide the new one. On stack_bowls
  it raised the mean score from 0.09 to 0.12.
- `AGENTICWAM_CODE_PIN=1` keeps the first program instead of regenerating it every replan.
- `ROBODOJO_RUN_ID` resumes a run; a new id starts fresh.
- `OMNI_KIT_ACCEPT_EULA=YES` had the user's go-ahead on d3. On a new machine, ask before setting it.

**Results.** They go under `RoboDojo/eval_result/RoboDojo/$TASK/AgenticWAM/arx_x5/*/$R/`:
- `_result.json` with per-episode score and success
- `episode_*_cam_head_*.mp4`, one frame per control step (`ROBODOJO_VIDEO_EVERY_STEP=1` is on by default for
  AgenticWAM)

**After the eval** (free):

```bash
RUN=$(ls -d ~/agentic-wam/RoboDojo/eval_result/RoboDojo/$TASK/AgenticWAM/arx_x5/*/$R)
$PY -I $K_DIR/state_compare.py $L/state.jsonl $L                       # eval vs training state at each policy call
$PY -I $K_DIR/rt_overlays.py $L/code.jsonl $RUN $L/videos <step>        # real-time video + generated program + replan counter
```

- `state_compare`: compare the nearest-training-state distance p95 against a held-out training episode (stack_bowls:
  3.0 vs 1.1). Look at the plot for gripper oscillation between calls and for joint branches never seen in training.
- `rt_overlays` writes `<step>_epNN.mp4` and `<step>_index.json`.
- For training-episode overlays, use `code_overlay.py` with segments at the dataset's own label boundaries, read
  per frame from `code_id_by_idx`. Never sample the labels on the 24-step replan grid: that hides where each label
  starts. Mark gripper flips from state dims 32 (L) and 33 (R).

**Report.**
- Per-episode scores, mean, and successes.
- How often each skill appears in the generated programs vs the training labels.
- Gripper flips between calls.
- The state-distance numbers.
- Offer the comparison page (Artifact) built from the overlay videos.
- Record settings and results in a new `CHOICES.md` section for the task.

## Known failure mode to look for (from stack_bowls)

**What happens.** The policy stacks one bowl, then loops: it re-grasps the same bowl and puts it back.

**Causes found.**
- **No history.** `code_memory: false`, no history in the code prompt, and one proprio token. Each call sees only the
  current frames and state.
- **Generated programs drift from the training labels.** `place` appeared in 11% of eval calls vs 39% of training
  frames.
- **Off-distribution IK.** IK from the near-singular all-zero home pose picks an elbow branch absent from training.

Check for these first on a new task before tuning anything else.
