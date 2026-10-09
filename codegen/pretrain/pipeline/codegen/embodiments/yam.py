"""YAM bimanual robot (arms + parallel grippers) pretraining source. API surface:
``codegen.api.bimanual``; prompts: ``codegen.actor.prompts_yam``; data: LeRobot datasets under
YAM_ROOT (``codegen.data.readers``)."""
from __future__ import annotations

from codegen.skill_paths import library_dir

from ._lerobot_source import LeRobotSourceSpec


class YamSpec(LeRobotSourceSpec):
    name = "yam"
    display_name = "YAM bimanual"
    skills_dir = library_dir("yam") / "skills"


SPEC = YamSpec()
