"""Code-execution environment base: sandboxed exec of generated/oracle programs against a registered API tier."""
import contextlib
import io
import sys
import traceback
from dataclasses import dataclass
from typing import Any, SupportsFloat

import numpy as np
from gymnasium import Env, spaces

from capsim.envs.base import BaseEnv, ObsType
from capsim.envs.configs.instantiate import instantiate as cfg_instantiate
from capsim.envs.configs.loader import DictLoader
from capsim.integrations.base_api import ApiBase, get_api


class Tee(io.TextIOBase):
    """Stream writes to several streams (console + capture buffer), so breakpoints still work."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


@dataclass
class CodeExecEnvConfig:
    """Configuration for a code-execution environment (``low_level`` is an env or a YAML path; ``apis`` are registered names)."""

    low_level: Env | str
    apis: list[str]
    prompt: str | None = None
    task_only_prompt: str | None = None
    multi_turn_prompt: str | None = None
    oracle_code: str | None = None
    privileged: bool = False
    enable_render: bool = True
    viser_debug: bool = False


class SimpleExecutor:
    """In-process executor exposing globals ``env``, ``APIS``, ``INPUTS`` and ``RESULT`` (imports unrestricted)."""

    def __init__(self, env: BaseEnv, apis: dict[str, ApiBase]) -> None:
        self._env = env
        self._apis = apis

    def run(self, code: str, *, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        g: dict[str, Any] = {
            "__name__": "__main__",
            "env": self._env,
            "APIS": self._apis,
            "INPUTS": inputs or {},
            "RESULT": None,
        }
        try:
            exec(code, g, g)
            return {"ok": True, "result": g.get("RESULT")}
        except BaseException as exc:  # defensive; propagate minimal info
            return {"ok": False, "error": repr(exc)}


class CodeExecutionEnvBase(Env):
    """High-level env that runs Python code and interacts with a low-level env."""

    prompt: str | None = None
    regenerate_prompt: str | None = None

    def __init__(self, cfg: CodeExecEnvConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.low_level_env: BaseEnv = self._build_low_level(
            cfg.low_level, cfg.privileged, cfg.enable_render, cfg.viser_debug
        )  # type: ignore[assignment]
        self._apis: dict[str, ApiBase] = {n: get_api(n)(self.low_level_env) for n in cfg.apis}
        self._executor = SimpleExecutor(self.low_level_env, self._apis)
        self._step_count = 0
        self.action_space = spaces.Text(max_length=4096)
        self.observation_space = spaces.Dict({"task_prompt": spaces.Text(max_length=4096)})

        # A YAML prompt overrides the class default.
        self._task_prompt = cfg.prompt if cfg.prompt is not None else self.prompt

        # oracle_code is a file path or inline code.
        if cfg.oracle_code is not None:
            import os

            self.oracle_code = (
                open(cfg.oracle_code, encoding="utf-8").read()
                if os.path.exists(cfg.oracle_code)
                else cfg.oracle_code
            )
        self._system_prompt = (
            "You are a helpful assistant that generates Python code to directly solve the task."
        )
        self._full_prompt = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": [{"type": "text", "text": self._get_complete_prompt()}]},
        ]

        # Variables persist across steps within an episode.
        self._exec_globals: dict[str, Any] = {}
        self._init_exec_globals()

    def compute_reward(self) -> float:
        """Reward of the current state, delegated to the low-level env."""
        return self.low_level_env.compute_reward()

    # ---- Private methods ----
    def _get_complete_prompt(self) -> str:
        """Task prompt followed by the APIs' combined docs."""
        docs = []
        for _name, api in self._apis.items():
            text = api.combined_doc()
            docs.append(f"\n{text.strip()}")
        return f"{self._task_prompt}\nAPIs:\n" + "\n".join(docs)

    def _exec_user_code(self, code: str) -> dict[str, Any]:
        obs = self._get_observation()
        self._exec_globals["obs"] = obs
        self._exec_globals["env"] = self.low_level_env
        self._exec_globals["APIS"] = self._apis
        for api in self._apis.values():
            for fn_name, fn in api.functions().items():
                self._exec_globals[fn_name] = fn

        stdout_buffer = io.StringIO()
        tee_out = Tee(sys.stdout, stdout_buffer)
        stderr_buffer = io.StringIO()
        tee_err = Tee(sys.stderr, stderr_buffer)
        ok = True
        try:
            with (
                contextlib.redirect_stdout(tee_out),
                contextlib.redirect_stderr(tee_err),
            ):
                exec(code, self._exec_globals, self._exec_globals)
        except BaseException:  # defensive; propagate minimal info
            ok = False
            traceback.print_exc(file=tee_err)

        return {
            "ok": ok,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "result": self._exec_globals.get("RESULT"),
        }

    def _init_exec_globals(self) -> None:
        """Reset the persistent exec namespace (called on construction and reset so episodes do not leak)."""
        g: dict[str, Any] = {
            "__name__": "__main__",
            "env": self.low_level_env,
            "APIS": self._apis,
            "INPUTS": {},
            "RESULT": None,
        }
        for api in self._apis.values():
            for fn_name, fn in api.functions().items():
                g[fn_name] = fn
        self._exec_globals = g

    def _build_low_level(
        self, src: Env | str, privileged: bool = False, enable_render: bool = True, viser_debug: bool = False
    ) -> BaseEnv:
        """Return ``src`` if it is an env, else instantiate it from a YAML path with ``_target_``."""
        if isinstance(src, str):
            if src.endswith(".yaml") or src.endswith(".yml"):
                cfg = DictLoader.load(src)
                if isinstance(cfg, dict) and "_target_" in cfg:
                    return cfg_instantiate(cfg)  # type: ignore[no-any-return]
                return cfg  # type: ignore[return-value]
            raise ValueError(f"low_level must be a yaml path (with _target_) or an env instance, got {src!r}")
        return src

    def _get_observation(self) -> dict[str, Any]:
        """Low-level observation plus ``full_prompt``."""
        obs = self.low_level_env.get_observation()
        obs.update({"full_prompt": self._full_prompt})
        return obs

    # ---- Public facing methods ----
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the low-level env and the exec namespace; returns (obs, info)."""
        self._step_count = 0
        obs, info = self.low_level_env.reset(seed=seed, options=options)
        obs.update(self._get_observation())
        self._init_exec_globals()
        self._exec_globals["INPUTS"] = obs
        info.update({"task_prompt": self._task_prompt})
        return obs, info

    def step(self, action: str) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute ``action`` as code; info carries sandbox_rc, stdout, stderr and task_completed."""
        self._step_count += 1
        exec_result = self._exec_user_code(action)
        obs = self._get_observation()
        # Sim substeps may skip viser updates; refresh so the view shows the final state.
        if hasattr(self.low_level_env, "viser_debug") and self.low_level_env.viser_debug:
            self.low_level_env._update_viser_server()
        reward = self.compute_reward()
        if hasattr(self.low_level_env, "task_completed"):
            task_completed = self.low_level_env.task_completed()
        else:
            task_completed = None
        terminated = reward == 1.0

        truncated = getattr(self.low_level_env, "_sim_step_count", 0) >= getattr(
            self.low_level_env, "max_steps", 999999
        )  # type: ignore[arg-type]

        if not exec_result["ok"] and exec_result["stderr"] == "":
            print("Uhh we shouldn't be here, sandbox return code 1 but stderr appears empty")
            raise RuntimeError("Sandbox return code 1 but stderr appears empty")

        info = {
            "sandbox_rc": 0 if exec_result["ok"] else 1,
            "stdout": exec_result["stdout"],
            "stderr": exec_result["stderr"],
            "task_prompt": self._task_prompt,
            "task_completed": task_completed,
        }
        return obs, reward, bool(terminated), bool(truncated), info

    def render(self, mode: str = "rgb_array"):
        return self.low_level_env.render(mode=mode)

    def render_wrist(self) -> np.ndarray | None:
        if hasattr(self.low_level_env, "render_wrist"):
            return self.low_level_env.render_wrist()
        return None

    def enable_video_capture(
        self,
        enabled: bool = True,
        *,
        clear: bool = True,
        wrist_camera: bool = False,
    ) -> None:
        import inspect

        sig = inspect.signature(self.low_level_env.enable_video_capture)
        if "wrist_camera" in sig.parameters:
            self.low_level_env.enable_video_capture(
                enabled, clear=clear, wrist_camera=wrist_camera
            )
        else:
            self.low_level_env.enable_video_capture(enabled, clear=clear)

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        return self.low_level_env.get_video_frames(clear=clear)

    def get_video_frame_count(self) -> int:
        if hasattr(self.low_level_env, "get_video_frame_count"):
            return self.low_level_env.get_video_frame_count()
        if hasattr(self.low_level_env, "_frame_buffer"):
            return len(self.low_level_env._frame_buffer)
        return 0

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        if hasattr(self.low_level_env, "get_video_frames_range"):
            return self.low_level_env.get_video_frames_range(start, end)
        if hasattr(self.low_level_env, "_frame_buffer"):
            return [f.copy() for f in self.low_level_env._frame_buffer[start:end]]
        return []

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        if hasattr(self.low_level_env, "get_wrist_video_frames"):
            return self.low_level_env.get_wrist_video_frames(clear=clear)
        if hasattr(self.low_level_env, "_wrist_frame_buffer"):
            frames = [f.copy() for f in self.low_level_env._wrist_frame_buffer]
            if clear:
                self.low_level_env._wrist_frame_buffer.clear()
            return frames
        return []

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        if hasattr(self.low_level_env, "get_wrist_video_frames_range"):
            return self.low_level_env.get_wrist_video_frames_range(start, end)
        if hasattr(self.low_level_env, "_wrist_frame_buffer"):
            return [f.copy() for f in self.low_level_env._wrist_frame_buffer[start:end]]
        return []


__all__ = [
    "SimpleExecutor",
    "CodeExecEnvConfig",
    "CodeExecutionEnvBase",
]
