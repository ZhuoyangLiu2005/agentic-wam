"""Shared spec for the two pretraining sources (``yam``, ``ego``): LeRobot datasets under a
configured root, read by ``codegen.data.readers``. Subclasses set the name, display name and
skill library; everything else is identical."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"


class LeRobotSourceSpec:
    name = ""
    display_name = ""
    skills_dir: Path

    @property
    def noise_labels(self) -> frozenset:
        """Annotation labels discarded as noise (EXACT normalized match — see
        codegen.grounding.segment.sub_task_blocks), from the NOISE_LABELS config key."""
        from codegen import config
        raw = config.get("NOISE_LABELS") or ""
        return frozenset(s.strip().lower() for s in raw.split(",") if s.strip())

    def list_tasks(self) -> list[str]:
        from codegen.data.readers import list_datasets
        return list_datasets(self.name)

    def preflight(self, task: str, episode_index: int = 0) -> str | None:
        """Error string if the episode's raw inputs are unusable, else None."""
        from codegen.data.readers import camera, episode_paths, source_root
        src = source_root(self.name) / task
        if not (src / "meta" / "info.json").exists():
            return f"no meta/info.json under {src}"
        pq, mp4, _ = episode_paths(src, episode_index, camera(self.name))
        if not pq.exists():
            return f"no parquet at {pq}"
        if not mp4.exists():   # the actor needs frames; no silent skips
            return f"no video at {mp4} (check {self.name.upper()}_CAMERA)"
        return None

    def run_dir(self, task: str, tag: str,
                model_short: str = "gemini-3.1-pro", is_default_model: bool = True,
                episode_id: str = "episode_000") -> Path:
        # embodiment/dataset/model/episode/tag hierarchy
        return OUTPUTS / self.name / task / model_short / episode_id / (tag or "dev")

    def batch_dir(self, tag: str) -> Path:
        return OUTPUTS / self.name / f"batch_{tag}"

    def load_episode(self, task: str, episode_index: int = 0):
        from codegen.data.readers import load_episode
        return load_episode(self.name, task, ROOT, episode_index)

    def load_state(self, task: str, episode_index: int = 0) -> dict:
        """{T, per_arm, sub_id, text, instruction} — parquet only, no video."""
        from codegen.data.readers import load_state
        return load_state(self.name, task, ROOT, episode_index)
