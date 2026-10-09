"""Skill library for the AgenticWAM coding expert.

Two roles, mirroring CaP-X (capx/skills/library.py):
  1. A FIXED, semantic *reduced skill vocabulary* (REDUCED_SKILLS) that the code expert's
     skill-selection head predicts over. Each skill is a named, documented unit; its
     name + docstring is the text we embed into the shared skill-library representation
     (see skill_embeddings.py).
  2. An optional EVOLVING library (SkillLibrary) that can ingest skills extracted from
     successful trial code and promote frequently-reused ones (CaP-Agent0 style).

The point of the redesign: the expert should reason about the *observation* and pick a grounded
skill from this library (which sub-goal applies now), instead of paraphrasing the instruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SkillSpec:
    """A fixed, semantic skill the planner can select. args lists expected argument names."""

    id: int
    name: str
    signature: str
    docstring: str
    args: tuple = ()


# Fixed reduced skill vocabulary for dual-arm tabletop manipulation (RoboTwin-style).
# Order defines the skill_id; index 0 is the no-op/terminal. Keep this list append-only so
# trained skill ids stay stable.
REDUCED_SKILLS = (
    SkillSpec(0, "DONE", "done()",
              "The current sub-goal is complete; emit when no further action is needed this chunk."),
    SkillSpec(1, "DETECT", "detect(obj)",
              "Locate a named object in the current observation and bind it for later skills.",
              ("obj",)),
    SkillSpec(2, "GRASP", "grasp(obj, arm)",
              "Move arm to a grasp pose for obj and close the gripper. Precondition: obj detected, not held.",
              ("obj", "arm")),
    SkillSpec(3, "LIFT", "lift(obj, arm)",
              "Raise the object currently held by arm straight up to clear the surface.",
              ("obj", "arm")),
    SkillSpec(4, "PLACE", "place(obj, target, arm)",
              "Place the held obj onto/at target and release. Precondition: obj held by arm.",
              ("obj", "target", "arm")),
    SkillSpec(5, "INSERT", "insert(obj, container, arm)",
              "Place the held obj INSIDE container (e.g. into a box/drawer/shoe) and release.",
              ("obj", "container", "arm")),
    SkillSpec(6, "PUSH", "push(obj, target, arm)",
              "Push obj along the table toward target (or a direction) without grasping.",
              ("obj", "target", "arm")),
    SkillSpec(7, "PRESS", "press(obj, arm)",
              "Press/click straight down on obj (e.g. a bell or button), then retract.",
              ("obj", "arm")),
    SkillSpec(8, "OPEN", "open(obj, arm)",
              "Open an articulated obj (drawer/door/lid) by grasping its handle and pulling.",
              ("obj", "arm")),
    SkillSpec(9, "CLOSE", "close(obj, arm)",
              "Close an articulated obj by grasping its handle and pushing.",
              ("obj", "arm")),
    SkillSpec(10, "HANDOVER", "handover(obj, from_arm, to_arm)",
              "Transfer the held obj from from_arm to to_arm at a midline pose.",
              ("obj", "from_arm", "to_arm")),
    SkillSpec(11, "ALIGN", "align(obj, target, arm)",
              "Fine-adjust the held obj pose to align with target before placing/inserting.",
              ("obj", "target", "arm")),
    SkillSpec(12, "RECOVER", "recover(obj, arm)",
              "Recover from a failed grasp of obj: reopen, retreat, and re-approach.",
              ("obj", "arm")),
    SkillSpec(13, "VERIFY", "verify(condition)",
              "Check a success condition (e.g. obj inside container) and report progress.",
              ("condition",)),
)

NUM_SKILLS = len(REDUCED_SKILLS)
SKILL_NAMES = tuple(s.name for s in REDUCED_SKILLS)
NAME_TO_ID = {s.name: s.id for s in REDUCED_SKILLS}
ID_TO_NAME = {s.id: s.name for s in REDUCED_SKILLS}


def skill_text(spec: SkillSpec) -> str:
    """Canonical text we embed for a skill: name + signature + docstring."""
    return f"{spec.name} - {spec.signature}: {spec.docstring}"


def skill_catalog() -> str:
    """Render the fixed skill vocabulary for a planner/VLM prompt (id, signature, doc)."""
    lines = ["# Skill vocabulary (pick one skill id per step):"]
    for s in REDUCED_SKILLS:
        lines.append(f"  [{s.id}] {s.signature}  -> {s.docstring}")
    return "\n".join(lines)


@dataclass
class _ExtractedSkill:
    name: str
    code: str
    docstring: str
    occurrences: int = 0
    source_tasks: list = field(default_factory=list)
    promoted: bool = False


class SkillLibrary:
    """Fixed reduced vocabulary + optional evolving extracted skills (CaP-Agent0 style)."""

    def __init__(self) -> None:
        self.fixed = REDUCED_SKILLS
        self.extracted = {}

    def id_of(self, name: str) -> int:
        return NAME_TO_ID[name.upper()]

    def name_of(self, sid: int) -> str:
        return ID_TO_NAME[int(sid)]

    def spec(self, key) -> SkillSpec:
        if isinstance(key, str):
            key = NAME_TO_ID[key.upper()]
        return self.fixed[int(key)]

    def catalog(self) -> str:
        return skill_catalog()

    def texts(self) -> list:
        """Embeddable text per skill id (ordered)."""
        return [skill_text(s) for s in self.fixed]


if __name__ == "__main__":
    print(skill_catalog())
    print(f"\nNUM_SKILLS={NUM_SKILLS}")
    print("sample embed text:", skill_text(REDUCED_SKILLS[2]))
