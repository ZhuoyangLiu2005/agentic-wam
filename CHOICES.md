# Choices: trex tong-transfer → AgenticWAM

## Conversion (DATA.md step 1) — `utils/convert_trex_v30_to_v21.py`
- Source `data_local/trex_gateway_tong_transfer_sf_norawtac_20260820` (LeRobot v3.0) → output `trex_tong_transfer_aw76_v21` (v2.1, 1 shard, 200 episodes, 208,581 frames)
- cam_high = head_left: crop rows 0–280, cols 140–540 (280×400, the 224:320 aspect ratio), then resize to 224×320 (INTER_AREA)
- Wrist cameras: full frame resized to 224×320, no crop
- State/action: raw 76-D Tianji+Sharpa layout; arm poses from FK using T-Rex `VegaFK` (vega_1 URDF, L_ee/R_ee frames), torso fixed at (0.9, 1.57, 0.1), head at (0.28, 0, 0)
- tactile_force is not in the training view
- Small pilot first, then the full run on r6000

## Stats and embeddings (steps 2–4), on r6000
- Only GPUs 4–5 may be used
- `dataset_stats_eef64.json` via `utils/make_stats.py --transform tianji_sharpa` (stride 5, horizon 32)
- Cosmos-Reason1-7B instruction embeddings, max-len 128

## Code annotation (step 5)
- Route A, embodiment `tianji_sharpa`, frozen `ego` skill library (not modified)
- VLM: `gcp/google/gemini-3.1-pro-preview` at `https://inference-api.nvidia.com/v1/chat/completions` (config `codegen/pretrain/pipeline/configs/tianji_sharpa.yaml`, tag v1)
- Key: `~/agentic-wam/API-key.txt` holds `api: <key>`. Strip the `api: ` label before use; the file is chmod 600 and gitignored, and the key is passed only as `VLM_API_KEY`
- Labeling view `top` stream: head_left at full 360×640, no crop; tactile and wrist EEF in its state
- Free tactile dry run on all 200 episodes → paid 5-episode pilot → you approve → full 200
- After the pilot (matched_raw 0.40): run all 200 episodes unchanged at the default 8 s max span (`trex_tong_transfer_aw76_label`), then a second full run with a **15 s max span** (`--max-span-s 15`, view `trex_tong_transfer_aw76_label_s15`, which symlinks the same data); compare the two in an HTML page
- The proposed fixes (C1 cut offset, C3/C4 tool-aware prompts) are not applied
- Results (2026-10-23), 200/200 episodes for both: 8 s has 1167 slices, matched_raw 0.188, repaired 0.336; 15 s has 856 slices, matched_raw 0.176, repaired 0.345. On slices with a tactile event, matched_raw is 0.105 at 8 s and 0.287 at 15 s
- **Cache built from the 15 s run** (chosen 2026-10-23): `trex_tong_transfer_aw76_v21/policy_code_idx_trex_v1.pt` with `episode_order.json` beside it; 200 episodes, 208,581 frames, 402 codes, 0 missing. Comments are kept verbatim (no `--strip-comments`). `verify` reports ALIGNED
- **Cache id-0 fix** (2026-10-23): `codegen/posttrain/build_flat_code_cache.py` gave the first real block (`robot.pick('tong', 'right')`) id 0. The repo reads id 0 as "no code" (pretraining's builder reserves it for `""`), so `dump_code_gt.py` and `preflight_sft.py` reported 25,203 frames without code, and `merge_datasets.py` would fail its assert. Id 0 is now reserved for `""`, and I rebuilt the cache on r6000: 403 entries (402 codes plus the empty one), 0 frames on id 0. The code for every frame is identical to the old cache, `verify` reports ALIGNED, and the dump shows 0 frames with NO code. The old cache is kept at `data_local/annot_scratch/policy_code_idx_trex_v1.old.pt`. The old lines are left commented:

```diff
diff --git a/codegen/posttrain/build_flat_code_cache.py b/codegen/posttrain/build_flat_code_cache.py
index b85ef18..f6441c2 100644
--- a/codegen/posttrain/build_flat_code_cache.py
+++ b/codegen/posttrain/build_flat_code_cache.py
@@ -48,8 +48,13 @@ def _canonical(block, strip_comments):
 def assemble_flat_cache(per_episode_frame_codes, strip_comments=False):
     """episodes' per-frame block strings -> {codes(unique), code_id_by_idx(int32)}. Dedups by the
     (optionally comment-normalized) string."""
-    codes: list[str] = []
-    index: dict[str, int] = {}
+    # codes: list[str] = []
+    # index: dict[str, int] = {}
+    # Reserve id 0 for "" (no code), the convention of codegen/pretrain/build_code_cache_flat.py that
+    # dump_code_gt / preflight_sft / merge_datasets / build_subgoal_from_code read as "empty". Without
+    # it the first real block got id 0 and those tools reported it as missing.
+    codes: list[str] = [""]
+    index: dict[str, int] = {"": 0}
     ids: list[int] = []
     EMPTY = "def policy_chunk(obs, state, robot, skills):\n    return skills.visual_diff(obs, obs)\n"
     for frames in per_episode_frame_codes:
```
- Code length in SDAR-1.7B tokens: max 156; per frame p50 49, p95 118, p99 144
- `model.infer_code_tokens=128` (chosen; truncates about 3% of frames, 18 of 402 codes). Training and serving must use the same value

## Post-training setup
- Frozen weights on r6000: Wan2.2 VAE at `~/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`, Qwen3-VL-2B-Instruct at `~/models/Qwen3-VL-2B-Instruct` (the vision tower; the checkpoint holds only its merger)
- SDAR-1.7B weights are **not downloaded**: the pretrained checkpoint holds every code-expert tensor. `paths.code_model` points to `~/models/SDAR-1.7B-Chat-tok` (config, tokenizer and model code only), and the SDAR body is built from config when that folder has no weights. A folder with weights still loads them, so pretraining from scratch is unchanged (it has no AgenticWAM checkpoint to supply the code expert). The old lines are left commented:

```diff
diff --git a/src/agenticwam/models/experts/code_expert.py b/src/agenticwam/models/experts/code_expert.py
index f8b02f2..5820734 100644
--- a/src/agenticwam/models/experts/code_expert.py
+++ b/src/agenticwam/models/experts/code_expert.py
@@ -659,10 +659,28 @@ def _build_sdar_code_expert(
     so map model.layers -> MoT CodeBlocks exactly as the Qwen builder, keep SDAR's embed/lm_head/
     norm + its native <|MASK|> token, and set code_objective='blockdiff'. Vision reaches the code
     via MoT joint attention (code->video), so no vision encoder is needed here."""
-    from transformers import AutoModelForCausalLM, AutoTokenizer
-    logger.info("Loading SDAR code-expert backbone %s ...", model_name)
-    model = AutoModelForCausalLM.from_pretrained(
-        model_name, trust_remote_code=True, torch_dtype=torch_dtype, attn_implementation="sdpa")
+    # from transformers import AutoModelForCausalLM, AutoTokenizer
+    # logger.info("Loading SDAR code-expert backbone %s ...", model_name)
+    # model = AutoModelForCausalLM.from_pretrained(
+    #     model_name, trust_remote_code=True, torch_dtype=torch_dtype, attn_implementation="sdpa")
+    # Load SDAR's weights when `model_name` has them (pretraining from scratch needs them: no AgenticWAM
+    # checkpoint initialises the code expert). A config-only dir (config + tokenizer + modeling code) builds a
+    # random-init body instead -- only valid when foundation_ckpt then overwrites every code-expert tensor
+    # (post-training).
+    import glob, os
+    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
+    has_weights = bool(glob.glob(os.path.join(str(model_name), "*.safetensors"))
+                       or glob.glob(os.path.join(str(model_name), "pytorch_model*.bin")))
+    if has_weights or not os.path.isdir(str(model_name)):
+        logger.info("Loading SDAR code-expert backbone %s ...", model_name)
+        model = AutoModelForCausalLM.from_pretrained(
+            model_name, trust_remote_code=True, torch_dtype=torch_dtype, attn_implementation="sdpa")
+    else:
+        logger.warning("SDAR dir %s has no weights: building the code-expert body from config (random init); "
+                       "foundation_ckpt must supply every code-expert tensor", model_name)
+        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
+        model = AutoModelForCausalLM.from_config(
+            cfg, trust_remote_code=True, torch_dtype=torch_dtype, attn_implementation="sdpa")
     text_model = model.model
     qwen_layers = text_model.layers
     hidden_dim = qwen_layers[0].self_attn.q_proj.in_features
```
- Check: built from config, the SDAR expert has 456 tensors, and the checkpoint's `dit.mot.experts.code.*` has the same 456. None are missing or unexpected and no shapes differ, so no random init survives
- Paths: `configs/paths/r6000.yaml` (select it with `paths=r6000`)
- Environment (`~/venvs/agenticwam-data`): installed the `pyproject.toml` pins (torch, torchvision and transformers were already installed and unchanged), plus `flash-attn==2.8.3` (prebuilt cu12 torch2.7 cxx11abiTRUE cp310 wheel). SDAR's `SDARRMSNorm` calls flash-attn's Triton `rms_norm_fn`, so it is required either way
- SDAR model-code fix (in `~/models/SDAR-1.7B-Chat-tok/modeling_sdar.py`, a downloaded file, not repo code; the original is in `modeling_sdar.py.orig`). transformers 4.57 renamed `LossKwargs` to `TransformersKwargs`. The original line is left commented:

```diff
-from transformers.utils import LossKwargs, auto_docstring, can_return_tuple, is_torch_flex_attn_available, logging
+# from transformers.utils import LossKwargs, auto_docstring, can_return_tuple, is_torch_flex_attn_available, logging
+# transformers>=4.57 renamed LossKwargs -> TransformersKwargs (used only as a TypedDict base below)
+from transformers.utils import auto_docstring, can_return_tuple, is_torch_flex_attn_available, logging
+try:
+    from transformers.utils import LossKwargs
+except ImportError:
+    from transformers.utils import TransformersKwargs as LossKwargs
```

## Post-training environment setup (`~/venvs/agenticwam-data`)
Python 3.10.22, built with uv 0.13.4. The exact package versions are in `configs/env/agenticwam-data.freeze.txt` (110 packages).
```bash
# uv (skip if installed): curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/agentic-wam
~/.local/bin/uv venv --python 3.10 ~/venvs/agenticwam-data
PY=~/venvs/agenticwam-data/bin/python
# repo + pinned deps (torch 2.7.1+cu128, torchvision 0.22.1+cu128, transformers 4.57.x, deepspeed 0.18.5, ...)
~/.local/bin/uv pip install --python $PY --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match -e .
# flash-attn: required by SDAR's RMSNorm (prebuilt wheel for torch 2.7 / cu12 / cxx11abi TRUE / cp310)
~/.local/bin/uv pip install --python $PY \
  "flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
# or reproduce the exact versions instead of the two installs above:
#   ~/.local/bin/uv pip install --python $PY --extra-index-url https://download.pytorch.org/whl/cu128 \
#     --index-strategy unsafe-best-match -r configs/env/agenticwam-data.freeze.txt && ~/.local/bin/uv pip install --python $PY --no-deps -e .
# check
$PY -c "import torch, transformers, flash_attn, deepspeed; print(torch.__version__, transformers.__version__, flash_attn.__version__, deepspeed.__version__, torch.cuda.is_available())"
```
The SDAR `LossKwargs` patch to `modeling_sdar.py` (above) is also needed, and the transfer tar already includes the patched file.

## Post-training command (r6000)
Fill in your wandb key (https://wandb.ai/authorize) on r6000 only. Do not commit or copy this file once the key is filled in.
```bash
export WANDB_API_KEY=""          # <- paste your wandb API key between the quotes
cd ~/agentic-wam
D=$PWD/data_local/trex_tong_transfer_aw76_v21
SKIP_PREFLIGHT=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 PYTHON=$HOME/venvs/agenticwam-data/bin/python \
bash scripts/train_real_sharpa.sh $D $D/policy_code_idx_trex_v1.pt paths=r6000 model.infer_code_tokens=128 \
  num_epochs=1 data.train.episode_indices=$D/episodes_0_99.json
```
- 8 GPUs (2026-10-24): 4 per GPU, global batch 32, `learning_rate` unchanged at 1e-4
- `max_steps=null` removed (2026-10-24): the length is the task config's `max_steps: 100000`, and `num_epochs=1` is ignored while `max_steps` is set (`trainer.py` `_estimate_total_train_steps`)

## Deployment on vega + Sharpa (2026-10-24)
Branch `worktree-vega-sharpa-deploy`, commits `5b096b5`, `1958079`, `05c8697` (not pushed). Serve with `deploy/remote/server.py` through `deploy/launch_server.sh`. The older `deploy/deploy_server.py` stays YAM-only.
- **Opt-in flag**: Sharpa behaviour turns on only with `--embodiment sharpa` (`EMBODIMENT=sharpa` in `launch_server.sh`). The default `yam` works as before, and the flag is never inferred from the task. The server refuses to start if the flag and the task's transform disagree (`sharpa` needs `TianjiSharpaUnifiedEEFTransform`, `yam` needs `YamUnifiedEEFTransform`), or if `sharpa` is given a checkpoint that is not 64-D
- **Cameras** (`deploy/common/sharpa_obs.py`): the conversion recipe, byte-identical (pixel test). Head 360×640 → crop rows 0–280, cols 140–540 → 224×320 (INTER_AREA). Wrists: full 360×640 → 224×320 (INTER_AREA), no crop. Wire keys: `head_camera` = cam_high, `left_camera` = cam_left_wrist, `right_camera` = cam_right_wrist
- The generic stages that still run afterwards (`deploy_policy._build_obs_video`) leave these frames unchanged under the default `AGENTICWAM_EVAL_PREPROC=parity` and under `legacy` (measured difference 0). `parity_crop` would crop them again, so the server refuses it with `--embodiment sharpa`
- **Resolution check** (`deploy/common/resolution_guard.py`): every `/act` frame must be exactly 640×360 (W×H) RGB. Otherwise the server returns HTTP 400 naming the camera, the size received and the likely cause (rescaled, W/H swapped, RGBA, grayscale, other aspect ratio). Nothing is rescaled on the server. `/health` → `camera_resolution` shows the last size seen per camera and the number rejected
- **State/action** (`deploy/common/vega_eef64.py`): the robot sends 58-D `[L arm 7, L hand 22, R arm 7, R hand 22]` (the layout of the source `observation.state`/`action`) and gets back an absolute 58-D chunk in the same layout
  - In: FK → raw 76-D, the same layout as `pack76` in the converter
  - Out: the model's 64-D delta → `TianjiSharpaUnifiedEEFTransform.backward` (reused, not rewritten) → IK row by row, each seeded with the previous row. Hands pass through as absolute values, and the zero gripper dims are dropped
  - A row whose IK misses 1 mm is still used but counted as `ik_failures` in `/health`
- **Kinematics** (`utils/vega_fk_ik/vega_kinematics.py`, copied from T-Rex, so no T-Rex checkout is needed at run time)
  - FK = `VegaFK` (`utils/unpack_lerobot_to_raw_fk.py`). IK = `PinkLocalIK` + `build_reduced_bimanual_robot` (`hardware_code/teleop/`), with the same costs, 5 iterations and the daqp solver. Both arms are solved together, as upstream
  - Torso (0.9, 1.57, 0.1), head (0.28, 0, 0). The IK posture target is the teleop home pose from `hardware_code/config/default.yaml`: L `[0.84, 0.51, 0.37, -1.30, -0.65, -0.29, -0.03]`, R `[-0.84, -0.51, -0.37, -1.30, 0.65, 0.29, 0.03]`
  - `urdf/vega_1.urdf` is byte-identical to the conversion and teleop URDFs (Apache-2.0 license copied beside it). Only the kinematic tree is loaded: no meshes and no collision model (upstream IK does not use one either)
- `deploy/common/policy_runtime.py`: the old "`YamUnifiedEEFTransform` or `NotImplementedError`" block was **replaced, not left commented**; see `git diff main..worktree-vega-sharpa-deploy -- deploy/common/policy_runtime.py`. `deploy/remote/server.py` and `deploy/launch_server.sh` only gained the flag
- Not supported for Sharpa: real-time chunking (`committed`/`freeze` are ignored)
- Extra packages for the deploy venv (from T-Rex `hardware_code/requirements.txt`): `pin>=3.2`, `pin-pink`, `qpsolvers`, `daqp`
- Checked on the laptop: camera crop byte-identical; generic stages are a no-op; bridge tensor shapes and hand pass-through (with stubbed kinematics); resolution check (1 accepted, 6 rejected cases). **Not yet run**: the real FK/IK (pinocchio/pink are not installed there). Before the robot, run `python utils/vega_fk_ik/vega_kinematics.py` (FK→IK round trip) and a smoke start with a nonexistent `CKPT` in the deploy venv

## Deployment command (template)
Superseded for the tong-transfer checkpoint by `deploy/setup_server_env_sharpa.sh` + `deploy/launch_server_sharpa.sh` (see "How to run the deployment" below). Kept for reference.
`INFER_CODE_TOKENS` must be 128, as in training (the `launch_server.sh` default is 64). `INSTR` should be the dataset's task string. Do not commit the token.
```bash
cd ~/agentic-wam
D=$PWD/data_local/trex_tong_transfer_aw76_v21
export AGENTICWAM_DEPLOY_TOKEN=$(python3 -c "import secrets;print(secrets.token_urlsafe(24))")
CKPT=<run>/checkpoints/weights/step_XXXXXX.pt DATASET=$D INSTR="<task instruction>" \
TASK=posttrain_sharpa EMBODIMENT=sharpa INFER_CODE_TOKENS=128 GPU=<gpu> PYTHON=$HOME/venvs/agenticwam-data/bin/python \
bash deploy/launch_server.sh --override paths=r6000
```

## Sharpa client, milestone 1: replay dry-run (2026-10-25)
The server is not run on r6000. The client reaches it at `127.0.0.1:8712` through a port forward from whichever GPU host serves it.
- `deploy/robot/sharpa_client.py`: `SharpaPolicyClient` subclasses `AgenticWamPolicyClient`. It reuses the keepalive transport, auth, JPEG q90 encoding and latency stats, and replaces only the payload:
  - 58-D state in, `(H, 58)` absolute chunk out
  - camera keys `head_camera`/`left_camera`/`right_camera`
  - frames checked client-side to be raw 640×360 uint8, with no `resize_to`
  - `/health` must report `embodiment=sharpa`, `action_dim=58`
  - `committed`/`freeze` are never sent (no RTC for Sharpa)
- `deploy/robot/run_sharpa_replay.py`: open-loop replay of **episode 0** of the T-Rex v3.0 dataset. Nothing moves
  - Uses v3.0 rather than the converted v2.1 shard, because those videos are already 224×320
  - Rows executed per replan = the server's `replan_steps` from each `/act` reply (24 by default)
  - Reports per replan and for the whole episode, per body part:
    - `action_error` (executed rows vs recorded action)
    - `row0_vs_state`
    - `seam_jump` (previous chunk's last executed row → next chunk's row 0)
  - Writes a JSON report to `runs/sharpa_replay/`
- Chunk seams: hard cut for now. A cross-fade gets added only if `seam_jump` on a real checkpoint shows jumps
- Checked on the laptop against a stub server that uses the real `contract`/`obs_codec`/`resolution_guard`: 4 replans on episode 0, frames accepted, and `--guard-test` gets a 320×224 head frame rejected with HTTP 400. **Not yet run** against the real checkpoint server
- Run: `uv run --no-project --with av --with loguru --with pandas --with pyarrow --with opencv-python-headless python deploy/robot/run_sharpa_replay.py --token-file <token file> --guard-test`

## Sharpa client, milestone 2: live vega + Sharpa driver (2026-10-25)
- **Vendored** T-Rex `hardware_code/` (commit `2a891d3`, MIT) into `deploy/robot/vega/`, so no T-Rex checkout is needed (`vega/README.md`)
  - Files: `arm_hand_control.py`, `robot_descriptions.py`, `ik_utils.py`, the two camera receivers, and the Sharpa hand MJCF with the meshes it references (Apache-2.0, ~14 MB, in git)
  - Only the import lines and `project_root` changed
  - `dexcontrol`, `dexmate_urdf` and the Sharpa Wave SDK stay installed packages
- `deploy/robot/vega_robot.py` `VegaSharpaRobot` replaces YAM's `YamRobot`. Its setup is the same as `eval_trex_async.py`:
  - 300 Hz `full_robot_action_loop` + `SmoothingAndSafetyManager(ruckig_smoothing=False)`
  - arm speed capped at **0.4 × URDF velocity** (T-Rex eval value, kept)
  - URDF limit clip and collision fallback
  - reset = OMPL collision-planned move to the `default.yaml` home pose at 0.3 rad/s
- **Collision obstacles:** paper eval values, table 0.76, back wall 0.60, right wall 0.75 (`default.yaml` has table 0.64 for the dataset site). Override with `--robot-config`
- **Episode end = hold only:** the buffer is cleared and the arms keep the last target; reset by hand or with the next episode's start move
- `deploy/robot/sharpa_guard.py` checks each chunk at 30 Hz before any row is sent. Thresholds come from the 208k demo frames:

  | Check | Threshold | Action | Demo basis |
  |---|---|---|---|
  | arm \|row 0 − measured state\| | > **0.3** rad | reject + hold | \|action − state\| p99 0.28 |
  | arm step between rows | > **0.15** rad | reject + hold | demo max 0.135 |
  | arm step between rows | > **0.07** rad | counted only | demo p99 0.065 |
  | non-finite value | any | reject + hold | |
  | joint outside URDF limits | any | clipped + counted | |

- Speed cap check on the demos: 1.4% of frames move faster than 0.4 × URDF velocity (0.95% faster than 1.0×). In those frames the arm lags the chunk; the next chunk starts from the measured state
- `deploy/robot/run_sharpa_eval.py`: `--preview-only` / `--dry-run` / real run
  - sequential loop: holds during inference, executes the server's `replan_steps` rows (24) at 30 Hz
  - `--max-replans 60` ≈ 48 s per episode (demo episodes are ~45 s)
  - per-episode artifacts in `runs/sharpa_eval/<session>/ep_NNN/`: `code/replan_NNN.txt`, `.timing.json`, `obs/replan_NNN.jpg` (raw + model view), `steps.jsonl`, `result.json`
- **Generated code is printed** on the client at every replan; repeats collapse to one line, because `AGENTICWAM_CODE_PIN=1` keeps the step-0 program for the whole episode
- **Checked on the laptop:**
  - the vendored collision model loads (58 joints, 83 collision geometries, obstacles added)
  - a full live run on fake `dexcontrol`/`sharpa` with real ZMQ camera receivers, real OMPL reset and the stub server: commands reached the robot, 0% overruns
  - `--dry-run` sends nothing
  - guard unit cases pass
- **Not yet run** on the real robot or with the real checkpoint

## Deploy server for the tong-transfer checkpoint (2026-10-25)
- **Checkpoint** `checkpoints/tong-transfer-sharpa/step20000/model.pt` (`{"model", "step": 20000}`; downloaded from the Hugging Face hub). Saved beside it:
  - `dataset_stats_eef64.json`: copied from r6000 `data_local/trex_tong_transfer_aw76_v21/`, sha256 `2f8fc4c8…a598`
  - `config.yaml`: **reconstructed**, because the run directory that produced `step20000` is not on r6000 (r6000 `runs/` only holds the failed Oct 23 pilot). It was composed on r6000 from the 8-GPU training command. Against the pilot's saved config it differs only in `output_dir`, `max_steps` (100000 vs null) and the wandb name/mode, so the model and data sections are identical. Replace it with the real `<run_dir>/config.yaml` if the training machine still has it
- **Server scripts:**
  - `deploy/setup_server_env_sharpa.sh`
    - `setup` (the default): uv venv (Python 3.10), repo pins (`FREEZE=1` for the exact training versions), flash-attn, `pin pin-pink qpsolvers daqp`, frozen weights from the Hugging Face hub (SDAR without weights), the SDAR `LossKwargs` patch, and `configs/paths/sharpa_deploy.yaml`
    - `check`: imports, CUDA, weights, SDAR patch, checkpoint, stats sha256 and a 40-step FK→IK round trip
  - `deploy/launch_server_sharpa.sh`
    - serves with `posttrain_sharpa`, `--embodiment sharpa`, `infer_code_tokens=128` and the tong instruction
    - token in `runs/deploy_sharpa/token.txt` (mode 600)
    - writes the log, the generated code and the imagined video under `runs/deploy_sharpa/`
    - `smoke` starts the server with no checkpoint
- **IK finding** (first real FK/IK run, on the laptop: pin 4.1, pink 4.4). The server's `vega_kinematics.ik_both` takes 5 steps per row at gain 0.2, the teleop setting. Server-style chains on demo episodes 0–1 (seed = state, then walk row to row, 960 rows):

  | IK iterations | rows over 1 mm | median error | p95 error | max error | time per 32-row chunk (laptop CPU) |
  |---|---|---|---|---|---|
  | 5 | 31% | 0.69 mm | 2.46 mm | 12.6 mm | 21 ms |
  | 10 | 2% | 0.17 mm | 0.60 mm | 4.1 mm | 29 ms |
  | 20 | 0% | 0.02 mm | 0.07 mm | 0.44 mm | 56 ms |
  | 40 | 0% | — | 0.04 mm | 0.06 mm | 111 ms |

  - `vega_kinematics.py`'s own self-test fails at 5 steps (64 mm), because it seeds 0.1 rad off. With 40 steps it reaches 0.05 mm, so the libraries work. `check` uses the 40-step round trip
  - **Decision (2026-10-25): keep 5 iterations**, the T-Rex teleop setting. IK lag is accepted, and the server counts each miss in `/health` → `ik_failures`

## How to run the deployment (GPU server + robot station)
One machine: the **robot station** (the T-Rex teleop workstation that drives the vega arms, Sharpa hands and cameras) also runs the GPU server. It needs Linux and CUDA, and about 35 GB free: ~23 GB frozen weights + 10 GB checkpoint. The server listens on `127.0.0.1:8712` only, and the client finds it there and reads the server's token from `runs/deploy_sharpa/token.txt`, so no tunnel or token copy is needed. There are two Python environments: the deploy venv for the server and T-Rex's `hardware_code` env for the robot client.

### GPU server, once (on the station)
1. Check out this repo on branch `sharpa-client` on the station.
2. Environment and frozen weights (about 23 GB into `~/models`; set `VENV`/`MODELS_DIR` to change where):
   ```bash
   bash deploy/setup_server_env_sharpa.sh
   ```
3. Checkpoint and stats. From the laptop:
   ```bash
   rsync -a checkpoints/tong-transfer-sharpa/step20000/ <station>:<repo>/checkpoints/tong-transfer-sharpa/step20000/
   ```
   This copies `model.pt` (10 GB), `dataset_stats_eef64.json` and the reconstructed `config.yaml`. The stats can also come straight from r6000 `data_local/trex_tong_transfer_aw76_v21/`.
4. Verify, then run a model-free start:
   ```bash
   bash deploy/setup_server_env_sharpa.sh check    # every line ok
   bash deploy/launch_server_sharpa.sh smoke       # loads the model without weights; Ctrl-C after READY
   ```

### GPU server, each session (terminal 1, deploy venv)
```bash
GPU=0 bash deploy/launch_server_sharpa.sh
```
- Wait for `READY | task=posttrain_sharpa embodiment=sharpa action_dim=58`
- There must be **no** `load_checkpoint: N checkpoint tensors were REJECTED` line
- The token is in `runs/deploy_sharpa/token.txt`, where the client reads it; do not commit it
- Defaults: `127.0.0.1:8712` (`HOST=0.0.0.0` would expose it on the network), 6 denoise steps, 24 rows per replan. Override with `PORT` / `NIS` / `REPLAN` / `CKPT` / `INSTR`
- Server output in `runs/deploy_sharpa/`:
  - `server_*.log`
  - `videos/episode{N}_codes.jsonl` (every replan's code)
  - `videos/episode{N}_code.txt`
  - `videos/episode{N}_step0_gen.mp4` (the imagined future; `AGENTICWAM_SAVE_ALL_REPLANS=1` for every replan)

### Robot station, once
1. Same checkout as the server; the client uses only `deploy/`.
2. Use T-Rex's `hardware_code` env (Python 3.10, with `dexcontrol`, `dexmate_urdf` and the Sharpa Wave SDK). `deploy/robot/vega/README.md` lists the packages; the client adds none.
3. Edit `deploy/robot/vega/config/default.yaml`: a copy of T-Rex `hardware_code/config/default.yaml`, read by `vega_robot.VegaRobotConfig.from_yaml` and saved into each run's `session.json`.
   - It is the default for `run_sharpa_eval.py`. `--robot-config <yaml>` selects a different file in the same format
   - Sections used: `robot` (home pose, torso, head), `hands` (serials, interpolate), `cameras` (IPs, ports, 640×360), `control` (30 / 300 Hz, reset tolerance), `environment` (collision obstacles)
   - Set `environment.table_height` to the measured height − 0.04 m. It currently holds the paper-eval values: table 0.76, back wall 0.60, right wall 0.75 (T-Rex's own default is 0.64 with no walls)

### Robot station, each session (terminal 2, T-Rex `hardware_code` env)
1. Start the T-Rex camera senders, as for teleop: `camera/stream_sender_dexmate.py` on the Jetson and `camera/stream_sender_zed_box.py` on the ZED box.
2. Run in order, from the repo root, once the server prints READY. Go to the next step only when the current one looks right:
   ```bash
   I="There is a metal tong, a coiled small pack of plastic bags, and a red bowl on the table. Use your right hand to pick up the tong, move the tong's end to the pack of plastic bags, close the tong to lift it up, transport it to the red bowl, open the tong to drop the pack of plastic bags in the bowl, and place the tong down on the table."
   # a) what the model will see + camera image size check; nothing moves
   python deploy/robot/run_sharpa_eval.py --instruction "$I" --preview-only
   # b) full loop with the guard, nothing sent to the robot
   python deploy/robot/run_sharpa_eval.py --instruction "$I" --dry-run
   # c) real run, someone on the e-stop
   python deploy/robot/run_sharpa_eval.py --instruction "$I"
   ```
   - **a)** all three cameras are 640×360 (`OK` in the printout), the log shows `server resolution guard OK` (the server refused a 320×224 frame), and the bottom row of `preview_model_view.jpg` looks like the training crops
   - **b)** `guard rejects 0`, and `max_row0_rad` is well under 0.3
   - **c)**
     - the robot does a collision-planned move to the start pose, then waits for Enter
     - the generated code prints at each replan
     - `s` / `f` / `q` + Enter ends the episode, and the arms hold their last target
     - each episode is saved to `runs/sharpa_eval/<session>/ep_NNN/` (code, model view, steps, result)

