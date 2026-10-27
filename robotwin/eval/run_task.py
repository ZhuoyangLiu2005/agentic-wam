"""Evaluate one checkpoint on one RoboTwin task in one process (Hydra; root config configs/robotwin/sim_robotwin.yaml).

Links `RoboTwin/policy/<EVALUATION.policy_name> -> robotwin/deploy` (default name `agenticwam_policy`; a link of that
name pointing elsewhere is refused), then runs RoboTwin's official `script/eval_policy.py` through the episode-count adapter with the config overrides
and tees its log. robotwin/hybrid_eval/launch.py and robotwin/eval/benchmark.py build these calls for you.

Example:
  python robotwin/eval/run_task.py \
    ckpt=/path/to/step_030000.pt EVALUATION.task_name=handover_block gpu_id=0
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    explicit = _resolve_optional_path(cfg.EVALUATION.dataset_stats_path, base=PROJECT_ROOT)
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    for parent in list(ckpt_path.parents)[:4]:
        candidates.append((parent / "dataset_stats.json").resolve())
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Tried explicit EVALUATION.dataset_stats_path "
        "and checkpoint parent directories. Pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}")
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}")
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _ensure_policy_symlink(robotwin_root: Path, policy_source_dir: Path, policy_name: str) -> Path:
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {policy_root}")
    policy_target = policy_root / policy_name
    source_resolved = policy_source_dir.resolve()
    if not policy_target.exists() and not policy_target.is_symlink():
        try:
            policy_target.symlink_to(source_resolved, target_is_directory=True)
            return policy_target
        except FileExistsError:
            pass                                   # a parallel shard created it first; verify it below
    if policy_target.is_symlink():
        target_resolved = policy_target.resolve()
        if target_resolved != source_resolved:
            raise RuntimeError(
                f"Policy symlink conflict: {policy_target} -> {target_resolved}, expected -> {source_resolved}"
            )
        return policy_target
    raise RuntimeError(
        f"Path already exists and is not a symlink: {policy_target}. Please handle it manually."
    )


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


@hydra.main(version_base="1.3", config_path="../../configs", config_name="robotwin/sim_robotwin")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    policy_name = str(cfg.EVALUATION.policy_name)
    policy_source_dir = (PROJECT_ROOT / "robotwin" / "deploy").resolve()
    if not policy_source_dir.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {policy_source_dir}")
    _ensure_policy_symlink(robotwin_root=robotwin_root, policy_source_dir=policy_source_dir, policy_name=policy_name)

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    # [dataprep] keep the last TWO segments of output_dir when present so a caller can pass
    # <run_stamp>/<shard> and get one timestamped top dir per suite run (isolating the
    # append-mode episode*_codes.jsonl across runs); a bare basename still works.
    _parts = output_dir.parts
    run_ts = os.path.join(*_parts[-2:]) if len(_parts) >= 2 and _parts[-2] not in ("", "robotwin", ckpt_tag, "eval_shards") else output_dir.name   # "eval_shards" = hybrid launch.py layout, not a run stamp
    if output_dir.name == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    run_output_dir = (PROJECT_ROOT / "evaluate_results" / "robotwin" / ckpt_tag / run_ts)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_output_dir / (
        f"eval_{str(cfg.EVALUATION.task_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    robotwin_eval_base = (
        PROJECT_ROOT / "evaluate_results" / "robotwin" / ckpt_tag / run_ts / str(cfg.EVALUATION.task_name)
    )

    sim_cfg_path = (PROJECT_ROOT / "configs" / "robotwin" / "sim_robotwin.yaml").resolve()
    sim_task = HydraConfig.get().runtime.choices.get("task")
    dataset_stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)

    overrides: list[str] = []
    _append_override(overrides, "task_name", cfg.EVALUATION.task_name)
    _append_override(overrides, "task_config", cfg.EVALUATION.task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt_path))
    _append_override(overrides, "seed", cfg.seed)
    _append_override(overrides, "policy_name", cfg.EVALUATION.policy_name)
    _append_override(overrides, "instruction_type", cfg.EVALUATION.instruction_type)
    _append_override(overrides, "eval_num_episodes", cfg.EVALUATION.eval_num_episodes)
    _append_override(overrides, "sim_cfg_path", str(sim_cfg_path))
    _append_override(overrides, "sim_task", sim_task)
    # the policy re-composes the training config inside RoboTwin's process: hand it the weight locations
    # (`paths=<name>` / `paths.<key>=...`) this run was given, or it would build from configs/paths/default.yaml
    paths_overrides = [o for o in HydraConfig.get().overrides.task if o.split("=", 1)[0].split(".", 1)[0] == "paths"]
    if paths_overrides:
        _append_override(overrides, "sim_paths_overrides", "|".join(paths_overrides))
    _append_override(overrides, "eval_output_dir", str(robotwin_eval_base))
    _append_override(overrides, "mixed_precision", cfg.mixed_precision)
    _append_override(overrides, "device", cfg.EVALUATION.device)
    _append_override(overrides, "dataset_stats_path", str(dataset_stats_path))
    _append_override(overrides, "action_horizon", cfg.EVALUATION.action_horizon)
    _append_override(overrides, "replan_steps", cfg.EVALUATION.replan_steps)
    _append_override(overrides, "num_inference_steps", cfg.EVALUATION.num_inference_steps)
    _append_override(overrides, "sigma_shift", cfg.EVALUATION.sigma_shift)
    _append_override(overrides, "rand_device", cfg.EVALUATION.rand_device)
    _append_override(overrides, "timing_enabled", cfg.EVALUATION.timing_enabled)
    _append_override(overrides, "eval_video_log", cfg.EVALUATION.get("eval_video_log", True))
    _append_override(overrides, "skip_get_obs_within_replan", cfg.EVALUATION.skip_get_obs_within_replan)
    _model_cfg = cfg.get("model", None)
    if _model_cfg is not None:
        _append_override(overrides, "use_code_expert", _model_cfg.get("use_code_expert", False))
        _append_override(overrides, "code_model_name", _model_cfg.get("code_model_name", "mock"))

    cmd = [
        sys.executable, "-u", str(PROJECT_ROOT / "robotwin" / "eval" / "robotwin_entry.py"),
        "--config", f"policy/{policy_name}/deploy_policy.yml",
        "--overrides", *overrides,
    ]
    env = os.environ.copy()
    env["TASK"] = str(cfg.EVALUATION.task_name)  # per-task skill-prompt lookup inside the policy
    # AGENTICWAM_CUDA_VISIBLE_DEVICES selects the card(s) the eval subprocess sees;
    # gpu_id stays an int (hydra cannot parse "0,1").
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("AGENTICWAM_CUDA_VISIBLE_DEVICES", str(cfg.gpu_id))
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_file, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd, cwd=str(robotwin_root), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"RoboTwin evaluation failed with return code {return_code}. Log: {log_file}")
    print(f"Evaluation finished successfully. Log saved to: {log_file}")
    OmegaConf.save(config=cfg, f=str(run_output_dir / f"eval_config_{str(cfg.EVALUATION.task_name)}.yaml"))


if __name__ == "__main__":
    main()
