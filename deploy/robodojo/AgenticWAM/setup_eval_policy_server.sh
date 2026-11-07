#!/bin/bash
# Policy server for the AgenticWAM RoboDojo adapter (called by eval.sh; same 10 positional args as the other XPolicyLab
# policies). Runs XPolicyLab/setup_policy_server.py with the agentic-wam Python env.
#
# ckpt_name picks the checkpoint:
#   /abs/path/to/model.pt (or step_*.pt)   that file
#   /abs/dir                                <dir>/model.pt, else the newest <dir>/step_*.pt or <dir>/checkpoints/weights/step_*.pt
#   <name>                                  $AGENTICWAM_CKPT_ROOT/<name>/model.pt  (the HF layout: step15000/model.pt)
# policy_conda_env: a venv/conda prefix (has bin/python, e.g. ~/venvs/agenticwam-data) or a conda env name.
# Env: AGENTICWAM_CKPT_ROOT (default <agentic-wam>/checkpoints/robodojo-stack-bowls),
#      AGENTICWAM_DATASET   (a dir with the training run's dataset_stats_eef64.json; default data_local/robodojo_x5_stack_bowls,
#                            else deploy/robodojo/stats/stack_bowls),
#      AGENTICWAM_PATHS     (default r6000; configs/paths/<name>.yaml), AGENTICWAM_NIS, AGENTICWAM_REPLAN,
#      AGENTICWAM_INSTRUCTION (default deploy.yml: the stack_bowls string), AGENTICWAM_INFER_CODE_TOKENS (default 128).
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"       # XPolicyLab/policy/AgenticWAM (the symlink)
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
AGENTICWAM_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../.." && pwd)"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

ckpt_root="${AGENTICWAM_CKPT_ROOT:-${AGENTICWAM_ROOT}/checkpoints/robodojo-stack-bowls}"
newest() { find "$@" -maxdepth 1 -type f -name 'step_*.pt' 2>/dev/null | sort -V | tail -n 1; }
if [[ "${ckpt_name}" == /* && -f "${ckpt_name}" ]]; then
    checkpoint_path="${ckpt_name}"
elif [[ "${ckpt_name}" == /* ]]; then
    checkpoint_path="${ckpt_name}/model.pt"
    [[ -f "${checkpoint_path}" ]] || checkpoint_path="$(newest "${ckpt_name}" "${ckpt_name}/checkpoints/weights")"
else
    checkpoint_path="${ckpt_root}/${ckpt_name}/model.pt"
fi
if [[ ! -f "${checkpoint_path:-}" ]]; then
    echo "[SERVER] no checkpoint for ckpt_name=${ckpt_name} (looked for ${checkpoint_path:-<dir>/model.pt or step_*.pt})." >&2
    echo "[SERVER] download one: hf download xinyuan-eval/robodojo-stack-bowls step15000/model.pt --local-dir ${ckpt_root}" >&2
    exit 2
fi
# The server reads only <dataset>/dataset_stats_eef64.json. Default: the local training view if present, else the
# stack_bowls stats shipped in deploy/robodojo/stats/stack_bowls (a fresh clone has no data_local/).
default_dataset="${AGENTICWAM_ROOT}/data_local/robodojo_x5_stack_bowls"
[[ -f "${default_dataset}/dataset_stats_eef64.json" ]] || default_dataset="${AGENTICWAM_ROOT}/deploy/robodojo/stats/stack_bowls"
dataset="${AGENTICWAM_DATASET:-${default_dataset}}"
paths_cfg="${AGENTICWAM_PATHS:-r6000}"

if [[ -x "${policy_conda_env}/bin/python" ]]; then
    PY="${policy_conda_env}/bin/python"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
    PY=python
fi

action_dim=$(bash "${XPL_ROOT}/utils/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
echo -e "\033[33m[SERVER] policy=${policy_name} task=${task_name} ckpt=${checkpoint_path}\033[0m"
echo -e "\033[33m[SERVER] dataset=${dataset} paths=${paths_cfg} action_dim=${action_dim} python=${PY}\033[0m"
echo -e "\033[33m[SERVER] host=${policy_server_host} port=${policy_server_port} gpu=${policy_gpu_id}\033[0m"

extra=()
[[ -n "${AGENTICWAM_NIS:-}" ]] && extra+=(num_inference_steps="${AGENTICWAM_NIS}")
[[ -n "${AGENTICWAM_REPLAN:-}" ]] && extra+=(replan_steps="${AGENTICWAM_REPLAN}")
[[ -n "${AGENTICWAM_INSTRUCTION:-}" ]] && extra+=(instruction="${AGENTICWAM_INSTRUCTION}")   # the dataset's task string
[[ -n "${AGENTICWAM_INFER_CODE_TOKENS:-}" ]] && extra+=(infer_code_tokens="${AGENTICWAM_INFER_CODE_TOKENS}")

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${BENCH_ROOT}:${PYTHONPATH:-}" \
    "${PY}" -u "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            checkpoint_path="${checkpoint_path}" \
            dataset="${dataset}" \
            paths="${paths_cfg}" \
            "${extra[@]}"
