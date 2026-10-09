"""Where the frozen skill libraries live: ``codegen/pretrain/skill_library/<library>/``.

The libraries sit next to the engine, not inside it. Two downstream embodiments reuse a pretraining
library unchanged, so their labels share its vocabulary: ``yam_sft`` -> ``yam`` and
``tianji_sharpa`` -> ``ego``. Resolve by library, never by embodiment name (an embodiment-named
directory that does not exist yields an empty library and 0% skill adoption, silently).
"""
from __future__ import annotations

from pathlib import Path

LIBRARY_ROOT = Path(__file__).resolve().parents[2] / "skill_library"
LIBRARY_OF = {"yam": "yam", "ego": "ego", "yam_sft": "yam", "tianji_sharpa": "ego"}


def library_dir(embodiment: str) -> Path:
    """``skill_library/<library>`` for an embodiment (holds skills/, _attic/, CLAUDE.md, FROZEN.md)."""
    return LIBRARY_ROOT / LIBRARY_OF.get(embodiment, embodiment)
