"""Actor-facing API surfaces (one per embodiment) + shared plumbing.

Surfaces: ``bimanual.py`` (YAM, ``YamBimanualApi``) and ``ego.py``
(``EgoHandApi``), each with its ``*_contracts.py`` table; select at runtime
via ``codegen.api.active`` (CODEGEN_EMBODIMENT). Shared: ``base.py``
(ToolResult/GraspCandidate/combined_doc), ``contracts.py`` (Contract/
GripEffect dataclasses)."""
from .base import GraspCandidate, ToolResult
from .contracts import Contract, GripEffect

__all__ = ["Contract", "GraspCandidate", "GripEffect", "ToolResult"]
