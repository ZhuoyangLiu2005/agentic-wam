"""The ACTIVE embodiment surface — resolved ONCE per process.

Every vocabulary-dependent module (`program/calls`, `program/validate`,
`program/effects`, `library`, `actor/prompts`) imports the API surface from
HERE instead of naming a concrete embodiment, so the same pipeline code
annotates YAM two-arm robot data and ego (egocentric human-hand) data.

Selection is process-level via the ``CODEGEN_EMBODIMENT`` environment variable
(``yam`` | ``ego``, default ``yam``), set by the CLI entry points BEFORE any
codegen import. Process-level on purpose: each actor run is one subprocess with
ONE vocabulary; nothing ever mixes surfaces inside a process, and the episode
cache key namespaces itself automatically because API_DOC differs.
"""
from codegen import config

EMBODIMENT = config.embodiment()   # CODEGEN_EMBODIMENT env > config file > "yam"

if EMBODIMENT in ("yam", "yam_sft"):
    from .bimanual import (  # noqa: F401
        API_DOC,
        ARG_VOCAB,
        _API_SIG,
        _ARM_PARAM,
        gripper_effects,
        resolve_effects,
    )
elif EMBODIMENT in ("ego", "tianji_sharpa"):
    from .ego import (  # noqa: F401
        API_DOC,
        ARG_VOCAB,
        _API_SIG,
        _ARM_PARAM,
        gripper_effects,
        resolve_effects,
    )
else:
    raise RuntimeError(
        f"CODEGEN_EMBODIMENT={EMBODIMENT!r} — expected 'yam', 'ego', 'yam_sft' or 'tianji_sharpa'")
