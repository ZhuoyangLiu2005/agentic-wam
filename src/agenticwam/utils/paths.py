"""Read a default location from configs/paths/default.yaml for command-line tools outside Hydra."""
from __future__ import annotations

from pathlib import Path

PATHS_FILE = Path(__file__).resolve().parents[3] / "configs" / "paths" / "default.yaml"


def default_path(key: str) -> str:
    import yaml
    return str((yaml.safe_load(PATHS_FILE.read_text()) or {})[key])
