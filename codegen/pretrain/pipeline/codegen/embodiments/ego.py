"""Ego (egocentric human hands) pretraining source. API surface: ``codegen.api.ego``; prompts:
``codegen.actor.prompts_ego``; data: LeRobot datasets under EGO_ROOT (``codegen.data.readers``)."""
from __future__ import annotations

from codegen.skill_paths import library_dir

from ._lerobot_source import LeRobotSourceSpec


class EgoSpec(LeRobotSourceSpec):
    name = "ego"
    display_name = "egocentric human hands"
    skills_dir = library_dir("ego") / "skills"


SPEC = EgoSpec()
