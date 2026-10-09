# Real-world robots: YAM and Tianji + Sharpa

How to post-train AgenticWAM on a real robot and run it there. Two robots are supported, **YAM** (bimanual arms +
parallel grippers) and **Tianji + Sharpa** (bimanual arms + 22-DoF tactile hands). Both use the cross-embodiment
**unified 64-D EEF** action/state space and start from the same pretraining checkpoint. Preparing a dataset is
[DATA.md](DATA.md); code labels are [CODE_ANNOTATION.md](CODE_ANNOTATION.md); the environment is
[INSTALL.md](INSTALL.md).

## 0. At a glance

| | YAM | Tianji + Sharpa |
|---|---|---|
| hardware | 2 arms, 2 parallel grippers, 3 cameras (head + 2 wrist) | 2 arms, 2 Sharpa 22-DoF tactile hands, 3 cameras (head + 2 wrist) |
| raw state / action | 82-D | 76-D |
| transform to 64-D | `yam_unified_eef.YamUnifiedEEFTransform` | `tianji_sharpa_unified_eef.TianjiSharpaUnifiedEEFTransform` |
| 64-D blocks used | EEF + gripper (hand block zero) | EEF + hand (gripper slots zero) |
| data config | `configs/real_post_train/data/yam.yaml` | `configs/real_post_train/data/sharpa.yaml` |
| task config | `configs/real_post_train/task/posttrain_yam.yaml` | `configs/real_post_train/task/posttrain_sharpa.yaml` |
| code labels | `scripts/posttrain_annotation_yam.sh` (embodiment `yam_sft`, `yam` skill library) | `scripts/posttrain_annotation_sharpa.sh` (embodiment `tianji_sharpa`, `ego` skill library) |
| train | `bash scripts/train_real_yam.sh <dataset> <code_cache.pt>` | `bash scripts/train_real_sharpa.sh <dataset> <code_cache.pt>` |
| serve on the robot | `deploy/` (§4) | not included (§4.8) |

## 1. The robots

**YAM.** A standard YAM station: two follower arms with parallel grippers, teleoperated for data collection; cameras `cam_high` (top-down
head), `cam_left_wrist`, `cam_right_wrist`. The 82-D raw vector carries both EEF poses as 4×4 homogeneous matrices and
the two gripper openings (0 closed … 1 open); `YamUnifiedEEFTransform` maps it into the 64-D space and leaves the
22-D hand blocks at zero. On the robot the arms are commanded in joint space (6 joints + gripper per arm), so the
server converts between the robot's 14-D joints and the model's 64-D space with YAM forward/inverse kinematics
(`utils/yam_fk_ik/`, §4.4).

**Tianji + Sharpa.** Two Tianji arms carrying Sharpa dexterous hands (22 joint angles each, with tactile sensing);
same three camera roles. The 76-D raw vector is the two arm poses (4×4) and the two 22-D hand vectors;
`TianjiSharpaUnifiedEEFTransform` fills the EEF and hand blocks and leaves the gripper slots at zero. Tactile streams
are stored next to the shards and used only by the code-labeling tools (tactile sub-task splits), not by the model.

The exact layouts are in [DATA.md](DATA.md) §2. Both robots' actions are the same kind: the EEF part is a **delta**
from the chunk's first-frame pose, expressed in that frame; hand joints and grippers are **absolute**.

The cameras must be framed as in the training data (§4.4).

## 2. Data

Each downstream task is one dataset directory in the post-training (SFT) format: LeRobot shards
`train_shard_*/`, `dataset_stats_eef64.json`, `cosmos_text_embeds_cache/` and a flat code-label cache. The full
recipe — converting recordings, stats, text embeddings, code labels, checks — is [DATA.md](DATA.md). The robot only
changes three choices:

| | YAM | Tianji + Sharpa |
|---|---|---|
| `utils/make_stats.py --transform` | `yam` | `tianji_sharpa` |
| code-labeling embodiment | `yam_sft` — sub-tasks split on gripper events | `tianji_sharpa` — sub-tasks split on tactile contact; reads the labeling view named by `TIANJI_DATASET` ([CODE_ANNOTATION.md](CODE_ANNOTATION.md) §4) |
| data config | `data/yam.yaml` (raw 82) | `data/sharpa.yaml` (raw 76) |

## 3. Post-training

```bash
export HF_HOME=/path/to/hf_cache WANDB_API_KEY=<key>      # PYTHON=<env python> if not on PATH
bash scripts/train_real_yam.sh    <dataset> <dataset>/policy_code_idx_<name>.pt [hydra overrides]
bash scripts/train_real_sharpa.sh <dataset> <dataset>/policy_code_idx_<name>.pt [hydra overrides]
# e.g. bash scripts/train_real_yam.sh /data/fold_towels /data/fold_towels/policy_code_idx.pt model.infer_code_tokens=192
```

One recipe per robot, any task — the dataset is a parameter, not a config:

| setting | value (both robots) |
|---|---|
| initialisation | a pretraining checkpoint (`paths.pretrain_ckpt`, read as `foundation_ckpt` by the task config) |
| model | `configs/model/agenticwam_cosmos.yaml` — 3 experts (video, code, action) |
| steps | 100k, `batch_size` 4 per GPU, `gradient_accumulation_steps` 1, 8 GPUs, DeepSpeed ZeRO-1 |
| optimiser | lr 1e-4, cosine, weight decay 1e-2, grad-norm clip 1.0 |
| checkpoints / eval | `save_every` 20000 (weights only), `eval_every` 2000 |
| stream dropout | `CFG_DROPOUT=0.15,0.0` (code, video) |
| inference code canvas | `model.infer_code_tokens` 64 — set it to the length of your code labels, and serve with the same value |

The script runs `scripts/preflight_sft.py` first — the config composes, the data config uses this robot's transform,
every path exists, the code cache covers every frame, every served instruction has a text embedding, the GPUs are
idle — then launches accelerate + DeepSpeed. `DRY_RUN=1` stops after the checks and prints the launch command;
`NNODES`/`NODE_RANK`/`MASTER_ADDR`/`MASTER_PORT` make it multi-node; `NPROC`, `ZERO`, `RUN_ID` (resume) are the other
knobs. The run lands in `./runs/posttrain_<robot>_<dataset dir name>/<RUN_ID>/checkpoints/weights/step_XXXXXX.pt`.
The data configs hold out no episodes (`val_set_proportion: 0.0`), so the periodic eval runs on training episodes
([DATA.md](DATA.md) §7).

## 4. Running a checkpoint on YAM

The model runs on a GPU host; the robot station (no GPU) captures observations, sends them over HTTP and executes the
returned joint targets.

```
  YAM station (no GPU)                           GPU host
  +-----------------------------+   POST /act   +---------------------------------------------+
  | 3 cameras + 14-D joint state | ------------> | deploy/remote/server.py                     |
  | control loop + safety       |               |   common/policy_runtime.py                  |
  |                             |               |     14-D joints --FK--> raw 82-D            |
  |                             |               |     model: video + code + action co-denoise |
  |                             | <------------ |     64-D delta chunk --IK--> 14-D joints    |
  +-----------------------------+ [32 x 14] abs +---------------------------------------------+
```

| file | role |
|---|---|
| `deploy/launch_server.sh` | starts the server with the deployment settings (§4.1) |
| `deploy/remote/server.py` | HTTP server: token auth, keep-alive, one inference at a time, timing telemetry |
| `deploy/common/policy_runtime.py` | builds the model once from the training config; observation → absolute joint chunk |
| `deploy/common/yam_eef64.py` | 14-D joints ↔ 64-D bridge (FK in, IK out), on `utils/yam_fk_ik/` |
| `deploy/common/contract.py`, `obs_codec.py` | the wire format, validated on both ends; image decode |
| `deploy/common/fast_joint_infer.py` | compiled per-step denoise core (CUDA graphs) for lower latency |
| `deploy/robot/` | the station side: `agenticwam_client.py` (client), `obs_pipeline.py` (camera checks), `run_agenticwam_eval.py` (evaluation loop), `measure_link.py` (link/timing check) |
| `deploy/linkqual_{server,client}.py` | model-free link qualification: can the station reach the host, and how fast |

### 4.1 Start the server

```bash
export AGENTICWAM_DEPLOY_TOKEN=$(python3 -c "import secrets;print(secrets.token_urlsafe(24))")
CKPT=runs/posttrain_yam_<dataset>/<RUN_ID>/checkpoints/weights/step_100000.pt \
DATASET=<dataset> INSTR="<the task instruction>" INFER_CODE_TOKENS=<as trained> \
  bash deploy/launch_server.sh
```

`DATASET` must be the directory the checkpoint was post-trained on — it supplies the normalisation stats and the
data config — and `INFER_CODE_TOKENS` the training value of `model.infer_code_tokens`. Copy the token to the station
(`/tmp/agenticwam_token.txt` by default). Optional env: `GPU` (0), `PORT` (8712), `NIS` (6), `REPLAN` (24), `TASK`
(`posttrain_yam`), `PYTHON`, `HF_HOME`; extra arguments go to `deploy/remote/server.py`. Startup loads the checkpoint
and the Cosmos-Reason1 text encoder, then burns one warm-up inference so the robot never sees the slow first call;
wait for `READY`. Reach a remote GPU host from the station through a tunnel, e.g. `ssh -N -L 8712:localhost:8712 <gpu-host>`.

`launch_server.sh` sets the deployment defaults:

| env | value | effect |
|---|---|---|
| `AGENTICWAM_DEPLOY_DTYPE` | `bf16` | load the model in bf16 (the built-in default is fp32; bf16 halves load time at no measured inference cost) |
| `AGENTICWAM_FAST_JOINT` | `1` | compile the per-step denoise core (`fast_joint_infer.py`) |
| `AGENTICWAM_CODE_PIN` | `1` | keep the program generated at the episode's first call for the rest of the episode |
| `AGENTICWAM_NOISE_SEED` | `1234` | fixed sampler noise |

`deploy/remote/server.py` flags:

| flag | default | |
|---|---|---|
| `--ckpt` | required | a missing path builds from pretraining weights only — **smoke mode**, the actions are meaningless; the server logs it and `/health` reports `smoke_mode` |
| `--task`, `--dataset`, `--override` | `posttrain_yam`, —, — | compose the training config exactly as trained (`--override model.infer_code_tokens=<N>`) |
| `--dataset-stats` | the data config's `pretrained_norm_stats` | must be the stats the checkpoint trained with |
| `--num-inference-steps` | 10 | denoising steps per call; the main speed/quality knob |
| `--replan-steps` | 24 | reported to the client; how many rows it executes before the next call |
| `--action-horizon` | 32 | chunk length; keep the training value |
| `--instruction` | an example phrase | fallback when a request carries none; set it (`INSTR` in the launcher) |
| `--port`, `--host`, `--token` | 8712, `0.0.0.0`, `$AGENTICWAM_DEPLOY_TOKEN` | a non-loopback bind without a token is refused unless `--insecure` |
| `--max-wait-sec`, `--max-body-mb`, `--socket-timeout` | 5, 32, 30 | a second request waits `--max-wait-sec`, then gets 503 |
| `--video-log-dir`, `--no-warmup` | —, off | save the imagined future video; skip the warm-up |

### 4.2 Endpoints

| method | path | request → response |
|---|---|---|
| GET | `/ping` | → `{ok, protocol}` — unauthenticated liveness |
| GET | `/health` | → model and server info: `task`, `ckpt`, `smoke_mode`, `action_space` (`eef64`), `action_dim` (14), `model_action_dim` (64), `action_horizon`, `replan_steps`, `num_inference_steps`, `model_dtype`, `ik_failures`, … |
| POST | `/reset` | `{}` → `{ok, reset, episode}` — call at the **start of every episode** |
| POST | `/act` | see below |

Every request except `/ping` carries `Authorization: Bearer <token>`.

```
/act request   {protocol: 1, instruction: str,
                images: {head_camera|left_camera|right_camera: {format: "jpeg"|"png", data: <base64>}},
                state: [14], seq: int, obs_stamp_ns: int,
                committed: [[14], ...] (optional), freeze: int (optional)}
/act response  {action_chunk: [[14] x 32], horizon, action_dim: 14, replan_steps, action_type: "qpos_abs",
                instruction, code, seq, obs_stamp_ns, timing: {queue_ms, infer_ms, server_ms}}
```

`code` is the program the code expert generated for this call. `committed`/`freeze` are optional real-time-chunking
guidance (§4.5).

### 4.3 Observation and action contract

* **Cameras** — `head_camera` = `cam_high`, `left_camera` = `cam_left_wrist`, `right_camera` = `cam_right_wrist`;
  `H×W×3` uint8 RGB at the camera's native size. The server applies the training transform (resize the smallest side,
  center-crop to 320×224, scale to [-1, 1]), so the **aspect ratio decides the framing**: 4:3 (640×480) frames
  exactly as the 320×240 training recordings; 16:9 loses about a fifth of the width. `deploy/robot/obs_pipeline.py`
  checks this and previews what the model sees.
* **Camera placement must match the training rig.** It is the largest real-world source of failure, and nothing in
  software detects it.
* **State** — the robot's 14-D joints `[L joint 0-5, L gripper, R joint 0-5, R gripper]`, joints in radians,
  grippers 0 (closed) … 1 (open), as recorded.
* **Action** — `action_chunk[i]` is an **absolute** 14-D joint target in the same layout. Command it directly; do not
  add the state again.
* **Safety is the station's job.** The server applies no limits: clamp per-step joint deltas against the measured
  position, clip joint limits, hold position when no chunk is available.

### 4.4 From joints to the model and back

`deploy/common/yam_eef64.py` makes a 64-D checkpoint drive a joint-commanded arm:

1. **In:** FK on the 14-D state gives both EEF poses; with the grippers they form the raw 82-D vector the training
   transform expects.
2. **Model:** the processor converts to 64-D and normalises with the dataset's stats; the model co-denoises video,
   code and a 32-step action chunk; the chunk is de-normalised to 64-D: per arm an EEF delta (anchored at the
   chunk-start pose) and an absolute gripper.
3. **Out:** every row's EEF target = chunk-start pose ∘ delta (the exact inverse of the training transform, not a
   row-to-row chain); damped least-squares IK, seeded with the previous row's solution, gives the joints; grippers
   pass through.

A row whose IK misses the 1 mm / 1 mrad tolerance still returns its best solution and is counted (`ik_failures` in
`/health`, and a warning in the log). Many failures mean the model is asking for unreachable poses. The kinematics
use the station model in `utils/yam_fk_ik/` (FK matches the recorded EEF exactly; IK round trip ≈ 1 mrad —
`utils/yam_fk_ik/REPLAY_GUIDE.md`); `deploy/tests/validate_yam_eef64.py` checks the bridge against a recorded episode.

### 4.5 Inference settings

* **Denoising steps** (`NIS` / `--num-inference-steps`) — latency is roughly linear in it. 6 (the launcher) to 10.
* **Code pinning** (`AGENTICWAM_CODE_PIN=1`) — the program generated at the first call of an episode is reused for
  every later call, so the plan stays fixed for the episode; `/reset` clears it.
* **Real-time chunking** (`committed`, `freeze` in `/act`; on with `AGENTICWAM_FAST_JOINT`, off with
  `AGENTICWAM_RTC=0`) — for a continuously moving arm. The client sends the joint rows it has already committed to
  (up to 24) and how many of them the inference delay covers; the server converts them to 64-D, holds the first
  `freeze` rows fixed during denoising and guides the next ones with a weight decaying to zero, so the new chunk
  continues the one being executed. Inference-only: the model was not trained with it. `deploy/tests/validate_rtc.py`
  checks it on a GPU.
* **The sampler is stochastic.** Repeated calls on the same observation differ (about 0.05 rad mean absolute in
  joint space); never judge a checkpoint on one rollout.
* **Debugging** — `AGENTICWAM_SAVE_ALL_REPLANS=1` with `--video-log-dir` saves the imagined future of every call (not
  just the first per episode); `AGENTICWAM_ORACLE_CODE_FILE=<file>` injects a fixed program instead of generating one;
  `AGENTICWAM_ABLATE_DROP_CODE=1` runs the action without the code.

### 4.6 Control loop and timing

A chunk row `i` is the plan for `obs_time + i` (30 Hz). When it arrives, `L = latency × control_hz` rows are already
in the past, and the rest must last until the next chunk arrives, so a continuously moving loop needs

```
horizon − L ≥ L   ⇔   control_hz ≤ horizon / (2 × latency)
```

With a 32-row horizon and about one second per call, that is ~15 Hz, half the training rate. The recommended loop is
therefore **sequential**: capture → `/act` (the arm holds) → execute the chunk at 30 Hz → repeat. Because the arm is
still while the model runs, row 0 matches the measured state and no latency compensation is needed; the motion is at
the trained speed, stop-and-go. Executing fewer rows per call (`exec_rows`) gives fresher observations at the cost of
more stops. Real-time chunking (§4.5) is the alternative for continuous motion.

Two measured rules for the link:

* **Keep one HTTP/1.1 connection open.** A new connection per call cost 770 ms of network time per `/act` through a
  port forward, against 185 ms with one kept-alive socket; `deploy/robot/agenticwam_client.py` does this.
* **The model, not the network, is the budget.** Shrinking images further buys little and risks changing the framing;
  send native frames.

Measure the real numbers for your checkpoint and link before booking robot time:
`python deploy/robot/measure_link.py --token-file /tmp/agenticwam_token.txt --requests 12` (no robot needed — it
synthesises frames) reports latency percentiles and the control rate that closes.

### 4.7 Running an evaluation on the station

`deploy/robot/run_agenticwam_eval.py` is the station-side evaluation loop (it imports the station's own hardware,
safety and camera modules, which are not part of this repository; `agenticwam_client.py` and `obs_pipeline.py` are the
pieces that speak this repository's contract). Run it in three stages:

1. `--preview-only` — capture, save what the model will see, print the checks; the arms never move.
2. `--dry-run` — the whole loop including inference and the safety filter, without commanding the arms; its safety
   report tells you whether the policy's commands stay inside the envelope.
3. the real run.

Start each episode from a pose inside the training distribution (`start_pose`), not the mechanical home: from far
outside it the first action is a lunge. Use a pose from your own dataset — the built-in `training_mean` is the mean of
an earlier YAM dataset. Call `/reset` at every episode start (the loop does).

Each session (one server configuration) is written as:

```
evaluate_results/yam_realworld/<date>_<task>_<ckpt>/
├── session.json            checkpoint, task, dtype, steps, server flags, link measurement
├── ep_000/
│   ├── meta.json           instruction, times, operator, camera-rig notes
│   ├── result.json         verdict (operator's success / failure / aborted), duration, stop reason, safety stats
│   ├── steps.jsonl         per control step: state, action_raw, action_sent, clamped, timings
│   ├── code/replan_NNN.txt       the program generated at each call
│   ├── imagined/replan_NNN.mp4   the predicted future video (--video-log-dir on the server)
│   └── obs/replan_NNN_{head,left,right}.jpg   the frames sent
└── summary.csv             one row per episode
```

The three artifacts localise a failure: a wrong action with a sensible imagined video and program is an action
problem; a garbled imagined video is a perception / out-of-distribution problem. Where `action_raw` and `action_sent`
differ, the safety envelope intervened — an episode where they differ constantly does not evaluate the policy. Keep
failed episodes, and start a new session for any change of checkpoint, dtype or step count.

### 4.8 Tianji + Sharpa

Post-training is supported (§3); a deployment server is not included. The runtime's joint ↔ 64-D bridge is YAM's
kinematics, so `deploy/common/policy_runtime.py` refuses a checkpoint whose data config uses any other transform. To
serve a Tianji + Sharpa checkpoint, add a bridge for that robot alongside `yam_eef64.py`: FK from the arm joints to the
two arm poses (with the 22-D hand states, the raw 76-D vector) on the way in; on the way out, apply each row's EEF
delta to the chunk-start arm pose, solve the arm IK, and pass the absolute 22-D hand targets through.

## 5. Notes

* Serve with the **same** `--dataset`, `model.infer_code_tokens` and stats the checkpoint trained with; a mismatch
  silently corrupts the actions.
* The instruction must be one the checkpoint was trained on; a new phrasing is an out-of-distribution test.
* One robot per server: inference is serialised, a second in-flight request gets 503.
* The server runs bf16 through `launch_server.sh`; without it the runtime loads fp32 (same actions, slower load).
* `/health` reports what was actually loaded (`model_dtype`, `smoke_mode`, `ik_failures`) — check it before
  enabling the arms.
