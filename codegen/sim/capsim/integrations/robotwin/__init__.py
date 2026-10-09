"""Bridge to the RoboTwin simulator: task loading, per-episode reset, and a cwd guard.

RoboTwin task code reads ``./assets/...`` relative to the CWD, so every import and ``setup_demo`` runs
inside :func:`robotwin_cwd`; heavy imports are lazy so ``import capsim`` works without RoboTwin.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import yaml

from capsim.profiles import DEFAULT_EMBODIMENT, load_profile

# CAPX_ROBOTWIN_ROOT points at a clone with populated assets.
_here = os.path.dirname(os.path.abspath(__file__))
ROBOTWIN_ROOT = os.environ.get(
    "CAPX_ROBOTWIN_ROOT",
    os.path.normpath(os.path.join(_here, "..", "..", "third_party", "RoboTwin")),
)


def _ensure_on_path() -> None:
    """Put the RoboTwin root on ``sys.path`` so ``import envs.<task>`` works."""
    if ROBOTWIN_ROOT not in sys.path:
        sys.path.insert(0, ROBOTWIN_ROOT)


@contextlib.contextmanager
def robotwin_cwd():
    """Temporarily ``chdir`` into the RoboTwin root, restoring the previous CWD on exit."""
    prev = os.getcwd()
    try:
        os.chdir(ROBOTWIN_ROOT)
        yield
    finally:
        os.chdir(prev)


def get_embodiment_config(robot_file: str) -> dict[str, Any]:
    """Load an embodiment's ``config.yml`` (mirrors collect_data.get_embodiment_config)."""
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _load_instruction(task_name: str) -> str:
    """First template in ``description/task_instruction/<task>.json``, else the task name."""
    path = os.path.join(ROBOTWIN_ROOT, "description", "task_instruction", f"{task_name}.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for key in ("seen", "unseen"):
            templates = data.get(key)
            if isinstance(templates, list) and templates:
                first = templates[0]
                return first[0] if isinstance(first, list) and first else str(first)
    except Exception:
        pass
    return task_name.replace("_", " ")


@dataclass
class RoboTwinHandle:
    """Holds an instantiated RoboTwin task and the config needed to reset it."""

    task: Any
    task_name: str
    task_config: str
    args: dict[str, Any]
    instruction: str
    arm_dim: tuple[int, int] = (6, 6)
    _cur_seed: int | None = field(default=None, repr=False)

    def setup(self, seed: int, ep_num: int = 0, **overrides: Any) -> None:
        """Reset to a fresh episode for ``seed``; ``overrides`` replace task args for this call only."""
        self._cur_seed = seed
        with robotwin_cwd():
            self.task.setup_demo(now_ep_num=ep_num, seed=seed, is_test=True, **{**self.args, **overrides})
        self.task.set_instruction(instruction=self.instruction)
        # Tasks whose check_success() reads play_once-only state get it from the scene (task_shims).
        from capsim.integrations.robotwin import task_shims

        task_shims.install(self.task)

    def expert_placeholders(self) -> dict[str, str]:
        """Run the expert on the current scene and return its placeholder map; consumes the scene."""
        with robotwin_cwd():
            info = self.task.play_once()
        return {str(k): str(v) for k, v in dict((info or {}).get("info", {}) or {}).items()}

    def episode_descriptions(self, placeholders: dict[str, str], n: int = 100) -> dict[str, list[str]]:
        """Filled seen/unseen instruction paraphrases for one episode (at most ``n`` each)."""
        _ensure_on_path()
        from description.utils.generate_episode_instructions import generate_episode_descriptions  # type: ignore
        with robotwin_cwd():
            out = generate_episode_descriptions(self.task_name, [placeholders], max_descriptions=n)
        d = out[0] if out else {}
        return {"seen": list(d.get("seen", [])), "unseen": list(d.get("unseen", []))}

    def close(self) -> None:
        if hasattr(self.task, "close_env"):
            with contextlib.suppress(Exception):
                self.task.close_env()


def load_robotwin_task(
    task_name: str,
    task_config: str = "demo_clean",
    embodiment: str = DEFAULT_EMBODIMENT,
    *,
    camera_depth: bool = True,
    render_freq: int = 0,
    eval_mode: bool = True,
    planner: str | None = None,
    extra_args: dict[str, Any] | None = None,
) -> RoboTwinHandle:
    """Instantiate a RoboTwin task with its assembled config (mirrors ``script/collect_data.py``)."""
    _ensure_on_path()
    # envs module-level code reads ./assets relative to the CWD.
    with robotwin_cwd():
        from envs._GLOBAL_CONFIGS import CONFIGS_PATH  # noqa: E402

    # --- Load the task-config profile (demo_clean / demo_randomized / ...) ---
    config_path = os.path.join(ROBOTWIN_ROOT, "task_config", f"{task_config}.yml")
    with open(config_path, encoding="utf-8") as f:
        args: dict[str, Any] = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config

    # --- Resolve embodiment files (mirrors collect_data.main) ---
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def _embodiment_file(name: str) -> str:
        robot_file = _embodiment_types[name]["file_path"]
        if robot_file is None:
            raise ValueError(f"Missing embodiment files for '{name}'")
        return robot_file

    args["embodiment"] = [embodiment]
    args["left_robot_file"] = _embodiment_file(embodiment)
    args["right_robot_file"] = _embodiment_file(embodiment)
    args["dual_arm_embodied"] = True
    args["embodiment_name"] = str(embodiment)

    with robotwin_cwd():
        args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
        args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if planner is not None:
        args["left_embodiment_config"]["planner"] = planner
        args["right_embodiment_config"]["planner"] = planner

    arm_dim = (
        len(args["left_embodiment_config"]["arm_joints_name"][0]),
        len(args["right_embodiment_config"]["arm_joints_name"][1]),
    )

    # --- cap-x runtime overrides: headless, depth on, no data dumping ---
    args["render_freq"] = render_freq
    args["eval_mode"] = eval_mode
    args["save_data"] = False
    args["collect_data"] = False
    data_type = dict(args.get("data_type", {}) or {})
    data_type["rgb"] = True
    if camera_depth:
        data_type["depth"] = True
    data_type["qpos"] = True
    data_type["endpose"] = True
    args["data_type"] = data_type
    # save_freq also paces the video-frame hook (RoboTwinEnv._install_frame_hook), so keep it set.
    args["save_freq"] = args.get("save_freq") or int(load_profile(embodiment).recording.save_freq)
    args["eval_video_save_dir"] = None

    if extra_args:
        args.update(extra_args)

    # --- Instantiate the task class (RoboTwin class_decorator) ---
    _ensure_on_path()
    with robotwin_cwd():
        envs_module = importlib.import_module(f"envs.{task_name}")
        try:
            env_class = getattr(envs_module, task_name)
            task = env_class()
        except AttributeError as exc:
            raise ValueError(
                f"RoboTwin task '{task_name}' has no class named '{task_name}' in envs/{task_name}.py"
            ) from exc

    return RoboTwinHandle(
        task=task,
        task_name=task_name,
        task_config=task_config,
        args=args,
        instruction=_load_instruction(task_name),
        arm_dim=arm_dim,
    )


__all__ = [
    "ROBOTWIN_ROOT",
    "RoboTwinHandle",
    "load_robotwin_task",
    "get_embodiment_config",
    "robotwin_cwd",
]
