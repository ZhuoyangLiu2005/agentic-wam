"""Batch RoboTwin runs into <output_dir>/<task_config>/<task_name>/<model>/run/trial_* via the official launch.

Usage (from codegen/sim, after `source env.sh`):
    $ROBOTWIN_PY iterate/scripts/run_robotwin_batch.py \
        --args.base-config-path iterate/configs/priv_clean.yaml \
        --args.tasks <task> \
        --args.resume-idx 101 --args.total-trials 105
"""

import copy
import os
import sys
import traceback
from dataclasses import dataclass, field

import tyro
import yaml

from capsim.envs.launch import LaunchArgs
from capsim.envs.launch import main as launch_main


@dataclass
class RoboTwinBatchLaunchArgs:
    """Command-line arguments for automated RoboTwin batch execution."""

    base_config_path: str = "iterate/configs/priv_clean.yaml"

    tasks: list[str] = field(default_factory=lambda: ["blocks_ranking_size"])
    """RoboTwin task names (envs/<task>.py module/class names)."""

    task_config: str | None = None
    """Override the base config's task_config (demo_clean / demo_randomized)."""

    models: list[str] = field(
        default_factory=lambda: ["gcp/google/gemini-3.1-pro-preview"]
    )

    server_url: str = "http://127.0.0.1:8110/chat/completions"

    output_dir: str = os.environ.get("CODEGEN_SIM_AGENT_ROOT", "outputs/agentic") + "/baseline"

    temperature: float = 1.0
    max_tokens: int = 2048 * 10
    reasoning_effort: str = "medium"
    api_key: str | None = None
    resume_idx: int | None = None
    """Start seed (overrides the base config's resume_idx)."""
    total_trials: int | None = None
    """End seed (inclusive; passed to launch)."""
    num_workers: int | None = None
    record_video: bool | None = None
    debug: bool = False
    use_oracle_code: bool | None = None


def main(args: RoboTwinBatchLaunchArgs) -> None:
    if not os.path.exists(args.base_config_path):
        print(f"Error: Base config file not found: {args.base_config_path}")
        sys.exit(1)
    with open(args.base_config_path, encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    failed_runs = []
    total = len(args.models) * len(args.tasks)
    idx = 1
    for model in args.models:
        for task_name in args.tasks:
            config = copy.deepcopy(base_config)
            low = config["env"]["cfg"]["low_level"]
            low["task_name"] = task_name
            if args.task_config:
                low["task_config"] = args.task_config
            task_config = low.get("task_config", "demo_clean")
            if args.resume_idx is not None:
                config["resume_idx"] = args.resume_idx

            # launch inserts <model> before the last component: .../<task>/<model>/run
            config_dir_base = os.path.join(
                os.path.abspath(args.output_dir), task_config, task_name
            )
            config["output_dir"] = os.path.join(config_dir_base, "run")
            os.makedirs(config_dir_base, exist_ok=True)
            config_path = os.path.join(config_dir_base, "config.yaml")
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f)

            print(f"\n[{idx}/{total}] model={model} task={task_config}/{task_name}")
            launch_args = LaunchArgs(
                config_path=config_path,
                server_url=args.server_url,
                model=model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                api_key=args.api_key,
                total_trials=args.total_trials,
                num_workers=args.num_workers,
                record_video=args.record_video,
                output_dir=None,
                debug=args.debug,
                use_oracle_code=args.use_oracle_code,
            )
            try:
                launch_main(launch_args)
            except Exception as e:
                print(f"ERROR running {task_name} with {model}: {e}")
                traceback.print_exc()
                failed_runs.append((model, task_name))
            idx += 1

    if failed_runs:
        print(f"Batch completed with {len(failed_runs)} failures: {failed_runs}")
        sys.exit(1)
    print("Batch execution completed successfully.")


if __name__ == "__main__":
    tyro.cli(main)
