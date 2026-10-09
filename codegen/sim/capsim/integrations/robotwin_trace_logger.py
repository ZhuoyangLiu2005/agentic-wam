"""RoboTwin traced API wrappers; ``envs/trial.py`` saves each trial's trace.json via ``get_trace_logger()``."""

from __future__ import annotations

from capsim.envs.base import BaseEnv
from capsim.integrations.robotwin.base_privileged import (
    RoboTwinGripperPrivilegedApi,
)
from capsim.integrations.robotwin.gripper import RoboTwinGripperApi
from capsim.integrations.trace_logger import TracedApiMixin


class TracedRoboTwinGripperApi(TracedApiMixin, RoboTwinGripperApi):
    """RoboTwinGripperApi with execution trace logging."""

    def __init__(self, env: BaseEnv) -> None:
        RoboTwinGripperApi.__init__(self, env)
        self.__init_trace__()
        self._trace_logger.set_env(env)


class TracedRoboTwinGripperPrivilegedApi(TracedApiMixin, RoboTwinGripperPrivilegedApi):
    """RoboTwinGripperPrivilegedApi with execution trace logging."""

    def __init__(self, env: BaseEnv, **kwargs) -> None:
        RoboTwinGripperPrivilegedApi.__init__(self, env, **kwargs)
        self.__init_trace__()
        self._trace_logger.set_env(env)
