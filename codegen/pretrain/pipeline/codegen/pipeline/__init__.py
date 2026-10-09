"""Orchestration: the per-episode actor pipeline and the reconciliation gate."""
from .gate import DATA, ROOT, gate_task, run_dir_for

__all__ = ["DATA", "ROOT", "gate_task", "run_dir_for"]
