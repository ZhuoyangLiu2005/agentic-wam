"""Per-embodiment adapters — the ASPIRE ``robot_adapters/`` analog.

ONE spec object per embodiment carries everything the shared machinery must
not know: task discovery, output layout (run/batch dirs), data preflight, the
episode reader, the callable-skill library location, and the display name.
Shared code (`pipeline/episode`, `pipeline/gate`, `scripts/run_batch`)
takes a spec and never branches on the embodiment name; the only selection
seams in the codebase are this registry, ``codegen.api.active`` (the API
surface), and the prompt-module import in ``codegen.actor.prompts`` — one seam
per concern, embodiment CONTENT lives in per-embodiment files.
"""
from __future__ import annotations


def get_spec(name: str):
    if name == "yam":
        from .yam import SPEC
    elif name == "ego":
        from .ego import SPEC
    elif name == "tianji_sharpa":
        from .tianji_sharpa import SPEC
    elif name == "yam_sft":
        from .yam_sft import SPEC
    else:
        raise ValueError(f"unknown embodiment {name!r} (expected 'yam', 'ego', 'yam_sft' or 'tianji_sharpa')")
    return SPEC


def active_spec():
    """Spec of the process's ACTIVE embodiment (CODEGEN_EMBODIMENT env)."""
    from codegen.api.active import EMBODIMENT
    return get_spec(EMBODIMENT)
