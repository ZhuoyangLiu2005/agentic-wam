# AgenticWAM in the RoboDojo sim

An XPolicyLab adapter that evaluates an AgenticWAM checkpoint post-trained on RoboDojo (dual ARX X5,
`utils/convert_robodojo_v21_to_yam82.py` data, e.g. `xinyuan-eval/robodojo-stack-bowls`) in the RoboDojo simulator.

| file | role |
|---|---|
| `AgenticWAM/model.py` | XPolicyLab `Model`: crops each 480×640 camera to rows 16:464, area-resizes to 224×320 (the training frames), packs the 14-D joint state, runs `deploy/common/policy_runtime.py` with `embodiment="x5"`, returns 24 absolute joint rows per chunk |
| `AgenticWAM/deploy.py` | episode loop: one observation per chunk, then the chunk's rows |
| `AgenticWAM/deploy.yml` | serving settings (instruction, `infer_code_tokens` 128, 6 denoising steps, replan 24, `paths`) |
| `AgenticWAM/eval.sh`, `setup_eval_env_client.sh` | XPolicyLab's standard launchers (copied from FastWAM) |
| `AgenticWAM/setup_eval_policy_server.sh` | resolves the checkpoint and the training dataset, starts the policy server in the agentic-wam env |
| `install.sh` | symlinks `AgenticWAM/` to `RoboDojo/XPolicyLab/policy/AgenticWAM`, installs the websocket server deps into the agentic-wam env |
| `../common/x5_eef64.py`, `../../utils/x5_kinematics.py` | X5 FK on the way in (identical to the training data), damped-least-squares IK on the way out |

## Run

```bash
cd ~/agentic-wam
bash deploy/robodojo/install.sh                      # once
hf download xinyuan-eval/robodojo-stack-bowls --local-dir checkpoints/robodojo-stack-bowls   # step{5000,10000,15000,20000}/model.pt

cd RoboDojo/XPolicyLab/policy/AgenticWAM
# eval.sh <bench> <task> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu> <env_gpu> <policy_env> <sim_conda_env>
EVAL_NUM=1 bash eval.sh RoboDojo stack_bowls step15000 arx_x5 joint 0 0 0 ~/venvs/agenticwam-data RoboDojo   # smoke: 1 episode
for s in 5000 10000 15000 20000; do
  bash eval.sh RoboDojo stack_bowls step$s arx_x5 joint 0 0 1 ~/venvs/agenticwam-data RoboDojo > eval_step$s.log 2>&1
done
```

* `ckpt_name`: a name under `$AGENTICWAM_CKPT_ROOT` (default `checkpoints/robodojo-stack-bowls`, read as
  `<name>/model.pt`), an absolute `.pt`, or an absolute training run dir (newest `step_*.pt`).
* `action_type` must be `joint`: the server solves IK itself. GPUs can be indices or `GPU-<uuid>`; the policy
  (model + Cosmos-Reason1-7B text encoder) and the sim fit on one 48 GB GPU.
* `EVAL_NUM=<n>` caps the episode count (default: the task's own count). Results and videos:
  `RoboDojo/eval_result/RoboDojo/stack_bowls/AgenticWAM/arx_x5/...`.
* Env: `AGENTICWAM_DATASET` (training dataset dir, default `data_local/robodojo_x5_stack_bowls`: its
  `dataset_stats_eef64.json` normalises state/action exactly as in training), `AGENTICWAM_PATHS` (default `r6000`,
  must give `text_encoder`, `wan_vae`, `code_model`, `code_vision`), `AGENTICWAM_NIS`, `AGENTICWAM_REPLAN`.
* The instruction is fixed to the training task string ("Stack the three bowls together."); the sim's own
  instruction text is ignored, since the checkpoint saw only that one.
* Isaac Sim asks to accept the NVIDIA Omniverse EULA on first start. A non-interactive run reads EOF and the client
  exits ("Unable to bootstrap inner kit kernel"): accept it once interactively, or set `OMNI_KIT_ACCEPT_EULA=YES` if
  you accept the license.
