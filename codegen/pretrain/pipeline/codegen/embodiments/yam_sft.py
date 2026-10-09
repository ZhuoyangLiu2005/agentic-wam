"""YAM post-training embodiment (`yam_sft`): YAM gripper data as LeRobot shards.

YAM is the engine's native embodiment (api/bimanual + prompts_yam + transitions.py detector + the
frozen yam skill library), so this spec adds NO new surface: it points task discovery at the
LeRobot view named by YAM_SFT_DATASET and takes sub-task ids from a per-shard `subtasks.json` sidecar
(gripper splits + VLM naming, written by codegen/posttrain/annotate_subtasks_gripper.py).

A "task" here is one TRAIN SHARD (train_shard_K); episodes are its local indices. Units are
`train_shard_K:<idx>`. Video = the shard's own cam_high stream; point YAM_SFT_DATASET at a
higher-resolution copy of the same episodes if the VLM cannot resolve task detail at 224x320.

The frozen yam skill library is reused unchanged: `codegen.skill_paths` maps yam_sft -> yam.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from codegen.skill_paths import library_dir
ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

def dataset_dir() -> Path:
    """The LeRobot view this embodiment labels (YAM_SFT_DATASET, required)."""
    v = os.environ.get("YAM_SFT_DATASET")
    if not v:
        raise RuntimeError("set YAM_SFT_DATASET to the LeRobot view to label (docs/CODE_ANNOTATION.md §4)")
    return Path(v)


def load_yam_sft_episode(shard: str, episode_index: int = 0):
    """Shard-local LeRobot read -> Episode. Mirrors readers._load_yam except sub_id/text, which
    come from <shard>/subtasks.json (there is no annotation.sub_task column downstream)."""
    import pandas as pd
    from codegen.data.readers import _YAM_ARM, Episode
    from codegen.grounding.transitions import debounce_events, grip_transitions

    d = dataset_dir() / shard
    pq = d / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    mp4 = d / "videos" / "chunk-000" / "observation.images.cam_high" / f"episode_{episode_index:06d}.mp4"
    df = pd.read_parquet(pq)
    st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)

    per_arm = {}
    for side, off in _YAM_ARM.items():                     # ee 0/16, grip 32/33 == our converter
        pos = st[:, off["ee"]:off["ee"] + 16].reshape(len(df), 4, 4)[:, :3, 3].astype(np.float64)
        raw = grip_transitions(st[:, off["grip"]])
        events, dropped = debounce_events(raw, pos)
        per_arm[side] = {"pos": pos, "grip_events": events, "dropped": dropped}

    sp = d / "subtasks.json"
    if not sp.exists():
        raise FileNotFoundError(f"{sp} missing -- run codegen/posttrain/annotate_subtasks_gripper.py first")
    rec = json.loads(sp.read_text())[str(episode_index)]
    sub_id = np.asarray(rec["sub_id"], dtype=np.int32)
    if len(sub_id) != len(st):
        raise ValueError(f"{shard}:{episode_index} sub_id len {len(sub_id)} != T {len(st)}")

    instruction = rec.get("instruction", "")
    if not instruction:
        for line in (d / "meta" / "episodes.jsonl").read_text().splitlines():
            r = json.loads(line)
            if int(r["episode_index"]) == episode_index:
                instruction = r["tasks"][0]

    try:
        from codegen.data.frames import LazyClip
        clip = LazyClip(str(mp4))
    except Exception:
        import av
        c = av.open(str(mp4))
        clip = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
        c.close()

    return Episode(instruction=instruction, clip=clip, sub_id=sub_id,
                   text={int(k): v for k, v in rec["text"].items()},
                   per_arm=per_arm, fps=30.0,
                   meta={"dataset": str(d), "episode_index": episode_index,
                         "detector": "gripper-velocity-debounced", "source": "yam_sft"},
                   episode_id=f"episode_{episode_index:03d}")


class YamSftSpec:
    name = "yam_sft"
    display_name = "YAM bimanual gripper (post-training)"
    noise_labels = frozenset()
    #: the FROZEN yam library (codegen.skill_paths maps yam_sft -> yam)
    skills_dir = library_dir("yam_sft") / "skills"

    def list_tasks(self):
        return sorted((p.name for p in dataset_dir().glob("train_shard_*")),
                      key=lambda n: int(n.rsplit("_", 1)[1]))

    def preflight(self, task: str, episode_index: int = 0):
        d = dataset_dir() / task
        if not (d / "meta" / "info.json").exists():
            return f"{d}: no meta/info.json"
        sp = d / "subtasks.json"
        if not sp.exists():
            return f"{sp} missing -- run annotate_subtasks_gripper.py first"
        if str(episode_index) not in json.loads(sp.read_text()):
            return f"episode {episode_index} absent from {sp}"
        return None

    def run_dir(self, task: str, tag: str, model_short: str = "gemini-3.1-pro",
                is_default_model: bool = True, episode_id: str = "episode_000"):
        return OUTPUTS / "yam_sft" / task / model_short / episode_id / (tag or "dev")

    def batch_dir(self, tag: str):
        return OUTPUTS / "yam_sft" / f"batch_{tag}"

    def load_episode(self, task: str, episode_index: int = 0):
        return load_yam_sft_episode(task, episode_index)

    def load_state(self, task: str, episode_index: int = 0) -> dict:
        ep = self.load_episode(task, episode_index)
        return {"T": len(ep.sub_id), "per_arm": ep.per_arm, "sub_id": ep.sub_id,
                "text": ep.text, "instruction": ep.instruction}


SPEC = YamSftSpec()
