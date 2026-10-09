"""Low-level environment interface (BaseEnv). Envs are built from YAML `_target_` configs (capsim.envs.configs)."""
from typing import Any, SupportsFloat, TypeVar, abstractmethod

from gymnasium import Env

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseEnv(Env):
    """Gymnasium base class for low-level simulator environments."""

    privileged: bool = False
    max_steps: int = 999999

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset to an initial state; returns (obs, info)."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Apply a low-level control action; returns (obs, reward, terminated, truncated, info)."""
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> ObsType:
        """Current observation."""
        raise NotImplementedError

    @abstractmethod
    def compute_reward(self) -> SupportsFloat:
        """Reward of the current state."""
        raise NotImplementedError

    @abstractmethod
    def task_completed(self) -> bool:
        """Whether the task is completed."""
        raise NotImplementedError


__all__ = ["BaseEnv"]
