"""Embodiment profiles: ``load_profile(name)`` reads ``capsim/profiles/<name>.yaml`` (robot/scene/recording
constants) into a read-only attribute tree; code holds no such literals, so a new robot is one new yaml."""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

DEFAULT_EMBODIMENT = "aloha-agilex"
PROFILE_DIR = Path(__file__).resolve().parent
_DEPTH_DIVISORS = {"mm": 1000.0, "m": 1.0}   # depth_unit -> divisor that yields metres


class Profile:
    """Read-only attribute / item access over the profile mapping (nested mappings become Profiles)."""

    __slots__ = ("_data", "name", "path")

    def __init__(self, data: dict[str, Any], name: str = "", path: Path | None = None) -> None:
        object.__setattr__(self, "_data", dict(data))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "path", path)

    def __getattr__(self, key: str) -> Any:
        try:
            v = self._data[key]
        except KeyError:
            raise AttributeError(f"profile {self.name or '?'} has no key {key!r} (file {self.path})") from None
        return Profile(v, self.name, self.path) if isinstance(v, dict) else v

    __getitem__ = __getattr__

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("profiles are read-only")

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:
        return f"Profile({self.name!r}, {self._data!r})"

    def near_hand(self, position) -> str:
        """The arm whose side of the workspace a world point lies on ("left" | "right")."""
        axis = {"x": 0, "y": 1, "z": 2}[str(self.robot.arm_side_axis)]
        positive = float(position[axis]) >= 0.0
        return "right" if positive == bool(self.robot.right_arm_is_positive) else "left"

    @property
    def depth_divisor(self) -> float:
        """Divide a raw depth map by this to get metres."""
        return _DEPTH_DIVISORS[str(self.recording.depth_unit)]


def profile_path(name: str) -> Path:
    return PROFILE_DIR / f"{name}.yaml"


@functools.lru_cache(maxsize=None)
def load_profile(name: str = DEFAULT_EMBODIMENT) -> Profile:
    """The profile named ``name`` (cached). Raises FileNotFoundError naming the expected file when it is missing."""
    path = profile_path(name)
    if not path.is_file():
        known = sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"no embodiment profile {name!r}: expected {path} (known: {known})")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Profile(data, name, path)


def embodiment_of(env: Any) -> str:
    """The embodiment name an env carries (``RoboTwinEnv.embodiment``), else the default profile name."""
    return str(getattr(env, "embodiment", None) or DEFAULT_EMBODIMENT)


__all__ = ["DEFAULT_EMBODIMENT", "PROFILE_DIR", "Profile", "embodiment_of", "load_profile", "profile_path"]
