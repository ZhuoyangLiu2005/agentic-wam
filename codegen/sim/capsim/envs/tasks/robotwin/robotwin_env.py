"""High-level code-execution env for RoboTwin tasks (the task itself is chosen on the low-level ``RoboTwinTask``)."""

import inspect
import json
from pathlib import Path
from typing import Any

from capsim.envs.tasks.base import CodeExecutionEnvBase


def _robotwin_full_description(handle: Any, task_name: str | None) -> str | None:
    """RoboTwin's placeholder-free ``full_description`` for the task (names the object: task identity, not GT state)."""
    task = getattr(handle, "task", None)
    if task is None or not task_name:
        return None
    try:
        root = Path(inspect.getfile(type(task))).resolve().parent.parent  # envs/ -> root
        jp = root / "description" / "task_instruction" / f"{task_name}.json"
        if jp.exists():
            fd = json.loads(jp.read_text()).get("full_description")
            return (fd[0].upper() + fd[1:]) if fd else None
    except Exception:  # best-effort
        return None
    return None



class RoboTwinCodeEnv(CodeExecutionEnvBase):
    """RoboTwin code env that prepends the episode's task goal to the YAML prompt.

    The live instruction keeps ``{A}``/``{B}`` placeholders (only ``play_once`` fills them), so it falls back to
    ``full_description``.
    """

    def _scene_header(self) -> str:
        handle = getattr(self.low_level_env, "handle", None)
        goal = getattr(handle, "instruction", None) if handle is not None else None
        if goal and "{" in goal:
            task_name = getattr(self.low_level_env, "task_name", None)
            goal = _robotwin_full_description(handle, task_name) or goal
        return f"TASK GOAL: {goal}" if goal else ""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        obs, info = super().reset(seed=seed, options=options)
        self._materialize_program(info)

        header = self._scene_header()
        if header:
            text = f"{header}\n\n{self._get_complete_prompt()}"
            self._full_prompt = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": [{"type": "text", "text": text}]},
            ]
            obs["full_prompt"] = self._full_prompt
            # base reset overwrites task_prompt; keep the live instruction for logging.
            handle = getattr(self.low_level_env, "handle", None)
            info["robotwin_instruction"] = getattr(handle, "instruction", None)

        return obs, info

    def _materialize_program(self, info: dict[str, Any]) -> None:
        """Fill a template program's ``"{B}"``-style keys with this episode's object nouns (no-op without keys)."""
        tmpl = getattr(self, "_oracle_template", None)
        if tmpl is None:
            tmpl = self._oracle_template = getattr(self, "oracle_code", None) or ""
        low = self.low_level_env
        low.episode_code = None
        low.episode_nouns = None
        task = getattr(low, "task", None)
        if not tmpl or task is None:
            return
        from capsim.integrations.robotwin import placeholders as P

        if not P.template_slots(tmpl):
            return
        slots = P.object_slots(task)
        text, nouns = P.materialize(tmpl, slots)
        self.oracle_code = text
        low.episode_code = text
        low.episode_nouns = {k: {"noun": nouns[k], "asset": slots[k]} for k in nouns}
        info["materialized"] = low.episode_nouns
        print("[materialize] seed %s: %s" % (info.get("seed"), ", ".join(
            f"{k} -> {nouns[k]!r} ({slots[k]})" for k in nouns)), flush=True)


__all__ = ["RoboTwinCodeEnv"]
