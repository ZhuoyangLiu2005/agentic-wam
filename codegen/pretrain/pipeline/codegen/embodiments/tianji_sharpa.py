"""TIANJI + SHARPA bimanual dexterous-hand embodiment spec (post-training).

Reuses the ego API surface (`codegen.api.ego`) and prompts (`codegen.actor.prompts_ego`)
deliberately: both are dexterous-hand vocabularies, and the frozen ego skill library
(`skill_library/ego/skills`) was authored against exactly that surface, so every ego skill loads
unmodified.

What differs from ego:
  * DATA is one LeRobot labeling view named by TIANJI_DATASET.
  * CONTACT EVENTS come from real per-finger tactile, not a fingertip-aperture proxy.
  * SUB-TASK BLOCKS come from `subtasks.json` (tactile splits + VLM naming), because real-robot
    downstream data carries no human annotation. See annotate_subtasks_tactile.py.

The prompt text is ego's, written for HUMAN hands; the frozen library was tuned against it, so it
is shared rather than forked.
"""
from __future__ import annotations

import os
from pathlib import Path

from codegen.skill_paths import library_dir
ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

def dataset_dir() -> Path:
    """The one LeRobot labeling view this embodiment annotates (TIANJI_DATASET, required)."""
    v = os.environ.get("TIANJI_DATASET")
    if not v:
        raise RuntimeError("set TIANJI_DATASET to the LeRobot labeling view (docs/CODE_ANNOTATION.md §4)")
    return Path(v)


class TianjiSharpaSpec:
    name = "tianji_sharpa"
    display_name = "TIANJI bimanual + Sharpa dexterous hands (teleop, tactile)"
    #: no human annotator, so no annotator noise vocabulary to filter
    noise_labels = frozenset()
    #: the frozen ego library, shared unchanged (codegen.skill_paths maps tianji_sharpa -> ego)
    skills_dir = library_dir("tianji_sharpa") / "skills"

    def list_tasks(self) -> list[str]:
        # one dataset == one task here; episodes are the unit, tasks the label
        return [dataset_dir().name]

    def preflight(self, task: str, episode_index: int = 0) -> str | None:
        d = dataset_dir()
        if not (d / "meta" / "info.json").exists():
            return f"{d}: not a LeRobot dir (no meta/info.json)"
        sp = d / "subtasks.json"
        if not sp.exists():
            return (f"{sp} missing — run codegen/posttrain/annotate_subtasks_tactile.py "
                    "(tactile splits + VLM sub-task naming) before labelling")
        import json
        if str(episode_index) not in json.loads(sp.read_text()):
            return f"episode {episode_index} absent from {sp}"
        return None

    def run_dir(self, task: str, tag: str,
                model_short: str = "gemini-3.1-pro", is_default_model: bool = True,
                episode_id: str = "episode_000") -> Path:
        return OUTPUTS / "tianji_sharpa" / task / model_short / episode_id / (tag or "dev")

    def batch_dir(self, tag: str) -> Path:
        return OUTPUTS / "tianji_sharpa" / f"batch_{tag}"

    def load_episode(self, task: str, episode_index: int = 0):
        from codegen.data.tianji_sharpa_reader import load_tianji_episode
        return load_tianji_episode(dataset_dir(), episode_index)

    def load_state(self, task: str, episode_index: int = 0) -> dict:
        """{T, per_arm, sub_id, text, instruction} — no video decode."""
        ep = self.load_episode(task, episode_index)
        return {"T": len(ep.sub_id), "per_arm": ep.per_arm, "sub_id": ep.sub_id,
                "text": ep.text, "instruction": ep.instruction}

    def n_episodes(self) -> int:
        import json
        return len(json.loads((dataset_dir() / "subtasks.json").read_text()))


SPEC = TianjiSharpaSpec()
