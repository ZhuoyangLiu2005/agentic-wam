"""Replay ONE RoboTwin seed with a given program through the official launch path (the fix-loop debug tool).

Usage (from codegen/sim, after `source env.sh`; $PY = $ROBOTWIN_PY):
    $PY iterate/scripts/replay_trial.py \
        --code $CODEGEN_SIM_AGENT_ROOT/fix_loop/<task>/fix_code.py \
        --seed <debug_seed> \
        --config iterate/configs/priv_clean.yaml --task <task> \
        --output-dir $CODEGEN_SIM_VALIDATION_ROOT/replay/<task>
Reward is in the trial dir name: trial_<seed>_..._reward_1.000 = success.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import tyro
import yaml


@dataclass
class ReplayArgs:
    code: str
    """Path to the program to replay (executed as oracle code, no LLM)."""

    seed: int
    """RoboTwin seed / trial index to run."""

    config: str = "iterate/configs/priv_clean.yaml"
    """Base env config to derive the replay config from."""

    task: str | None = None
    """RoboTwin task name; overrides the config's low_level.task_name.
    Required when using the generic _base_* configs."""

    task_config: str | None = None
    """demo_clean | demo_randomized; overrides the config's low_level.task_config."""

    output_dir: str = os.environ.get("CODEGEN_SIM_VALIDATION_ROOT", "outputs/validation") + "/replay"
    """Where the trial dir is written."""

    record_video: bool = True


def main(args: ReplayArgs) -> None:
    from capsim.envs.launch import LaunchArgs
    from capsim.envs.launch import main as launch_main

    code_path = os.path.abspath(args.code)
    if not os.path.exists(code_path):
        raise FileNotFoundError(code_path)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["env"]["cfg"]["oracle_code"] = code_path  # tasks/base.py loads the file
    low = cfg["env"]["cfg"]["low_level"]
    if args.task:
        low["task_name"] = args.task
    if args.task_config:
        low["task_config"] = args.task_config
    if str(low.get("task_name", "")).startswith("PLACEHOLDER"):
        raise ValueError(
            f"{args.config} is a generic base config — pass --task <robotwin task>"
        )
    # Oracle replay is single-shot: a set multi-turn prompt would make trial.py query the LLM and hang.
    cfg["env"]["cfg"].pop("multi_turn_prompt", None)
    cfg["resume_idx"] = args.seed
    cfg["trials"] = args.seed
    cfg["num_workers"] = 1
    cfg["output_dir"] = os.path.abspath(args.output_dir)
    cfg["record_video"] = args.record_video

    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="robotwin_replay_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    try:
        launch_main(
            LaunchArgs(
                config_path=tmp,
                use_oracle_code=True,
                record_video=args.record_video,
                total_trials=args.seed,
                num_workers=1,
            )
        )
    finally:
        os.unlink(tmp)


if __name__ == "__main__":
    main(tyro.cli(ReplayArgs))
