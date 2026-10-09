"""Per-embodiment episode readers for the pretraining sources — ONE interface, no information filtered.

The pipeline (``codegen.pipeline.episode``) consumes exactly five things per
episode; the reader produces them from the source, so adding an embodiment
never touches the pipeline:

    instruction : str
    clip        : (T,H,W,3)-indexable frames (LazyClip: forward-only lazy decode)
    sub_id      : (T,) int per-frame sub-task id     -> segment.sub_task_blocks
    text        : {id: label}
    per_arm     : {side: {"pos": (T,3) float64, "grip_events": [(frame, kind)]}}

Both pretraining sources are LeRobot datasets in the standard layout
(docs/PRETRAINING.md §1): a source root holds one LeRobot dataset per
directory (``train_shard_0``, ``train_shard_1``, …), and a work unit is
``<dataset dir>:<episode_index>``. Paths come from each dataset's own
``meta/info.json`` templates, and frames decode on demand at the stored
resolution via ``codegen.data.frames.LazyClip``.

  yam  ``observation.state`` 82-D: [0:16] / [16:32] left / right EEF pose
       (4x4, row-major), [32] / [33] left / right gripper (1 open .. 0 closed).
       Contact events: gripper open/close transitions (``grounding.transitions``).
  ego  ``observation.state`` 76-D: [0:16] / [16:32] left / right wrist pose
       (4x4, row-major, world frame), [32:54] / [54:76] the 22-D hands.
       Contact events: the hand-aperture detector (``grounding.aperture``) on
       the optional per-frame fingertip keypoints column (EGO_KEYPOINTS,
       float[2*21*3] = left then right hand, 21 MediaPipe-ordered points);
       without it the episode carries no contact events.

Sub-task blocks, first match wins:
  1. a per-frame ``annotation.sub_task`` int column whose values index
     ``meta/tasks.jsonl`` (0 / "" = unlabeled);
  2. ``<dataset>/subtasks.json``: {"<episode_index>": {"sub_id": [T ints],
     "text": {"<id>": "<label>"}, "instruction": "..."}} — e.g. written by
     ``codegen/posttrain/annotate_subtasks_gripper.py``;
  3. otherwise the whole episode is one block labelled with its instruction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from codegen import config
from codegen.data.frames import LazyClip

#: per-embodiment config keys: dataset root and the camera stream the actor sees
SOURCES = {"yam": {"root": "YAM_ROOT", "camera": "YAM_CAMERA"},
           "ego": {"root": "EGO_ROOT", "camera": "EGO_CAMERA"}}

#: yam 82-D: per arm the EEF pose (row-major 4x4) offset and the gripper index
_YAM_ARM = {"left": {"ee": 0, "grip": 32}, "right": {"ee": 16, "grip": 33}}
#: ego 76-D: per hand the wrist pose (row-major 4x4) offset
_EGO_WRIST = {"left": 0, "right": 16}
_WRIST, _THUMB_TIP, _INDEX_TIP, _MIDDLE_TIP = 0, 4, 8, 12   # MediaPipe hand-landmark ids


@dataclass
class Episode:
    instruction: str
    clip: object                     # (T,H,W,3)-indexable frames: LazyClip (or ndarray)
    sub_id: np.ndarray               # (T,) int32
    text: dict                       # {id: label}
    per_arm: dict                    # {side: {"pos", "grip_events", ...}}
    fps: float = 30.0
    meta: dict = field(default_factory=dict)
    episode_id: str = "episode_000"  # identity segment of the run-dir path


def source_root(embodiment: str) -> Path:
    """The configured dataset root of a pretraining source, resolved on first use."""
    return config.get_path(SOURCES[embodiment]["root"])


def camera(embodiment: str) -> str:
    return config.get(SOURCES[embodiment]["camera"])


def list_datasets(embodiment: str) -> list[str]:
    """LeRobot dataset dirs under the source root, in numeric order (train_shard_2 < train_shard_10)."""
    base = source_root(embodiment)
    if not base.is_dir():
        return []

    def key(name: str):
        head, _, tail = name.rpartition("_")
        return (head, int(tail)) if tail.isdigit() else (name, -1)
    return sorted((d.name for d in base.iterdir()
                   if d.is_dir() and (d / "meta" / "info.json").exists()), key=key)


def episode_paths(src: Path, episode_index: int, cam: str) -> tuple[Path, Path, dict]:
    """(parquet, mp4, info) for ONE episode of a LeRobot dataset dir, from its own path templates."""
    info = json.loads((src / "meta" / "info.json").read_text())
    chunk = episode_index // int(info.get("chunks_size", 1000))
    pq = src / info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    ).format(episode_chunk=chunk, episode_index=episode_index)
    mp4 = src / info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    ).format(episode_chunk=chunk, episode_index=episode_index, video_key=cam)
    return pq, mp4, info


def _tasks_text(src: Path) -> dict:
    text = {}
    for line in (src / "meta" / "tasks.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            text[int(r["task_index"])] = r["task"]
    return text


def _instruction(src: Path, episode_index: int, default: str) -> str:
    """This episode's instruction from meta/episodes.jsonl, matched BY INDEX."""
    ej = src / "meta" / "episodes.jsonl"
    if ej.exists():
        for line in ej.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r.get("episode_index", -1)) == episode_index:
                return (r.get("tasks") or [default])[0]
    return default


def _sub_tasks(src: Path, df, episode_index: int, instruction: str):
    """(sub_id[T], {id: label}) — annotation column, else subtasks.json, else one block."""
    T = len(df)
    if "annotation.sub_task" in df.columns:
        return df["annotation.sub_task"].to_numpy().astype(np.int32), _tasks_text(src)
    sp = src / "subtasks.json"
    if sp.exists():
        rec = json.loads(sp.read_text()).get(str(episode_index))
        if rec is not None:
            sub_id = np.asarray(rec["sub_id"], dtype=np.int32)
            if len(sub_id) != T:
                raise ValueError(f"{sp} episode {episode_index}: sub_id length {len(sub_id)} != T {T}")
            return sub_id, {int(k): v for k, v in rec["text"].items()}
    return np.ones(T, dtype=np.int32), {1: instruction}


def _blocks(sub_id: np.ndarray, text: dict) -> list[tuple[int, int, str]]:
    """(start, end, label) runs of labelled sub-task ids — the label-boost pass needs them."""
    segs, i, T = [], 0, len(sub_id)
    while i < T:
        j = i
        while j < T and sub_id[j] == sub_id[i]:
            j += 1
        label = text.get(int(sub_id[i]), "")
        if int(sub_id[i]) != 0 and label:
            segs.append((i, j, label))
        i = j
    return segs


def _per_arm(embodiment: str, df, fps: float, sub_id: np.ndarray, text: dict) -> dict:
    st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    T = len(st)
    per_arm = {}
    if embodiment == "yam":
        from codegen.grounding.transitions import debounce_events, grip_transitions
        for side, off in _YAM_ARM.items():
            pos = st[:, off["ee"]:off["ee"] + 16].reshape(T, 4, 4)[:, :3, 3]
            # hover-chatter suppression needs the TCP, so it runs HERE rather than
            # inside the detector; GRIP_MIN_DWELL=0 (the default) leaves events untouched
            events, dropped = debounce_events(grip_transitions(st[:, off["grip"]]), pos)
            per_arm[side] = {"pos": pos, "grip_events": events, "dropped": dropped}
        return per_arm

    kp_col = config.get("EGO_KEYPOINTS")
    kp = None
    if kp_col and kp_col in df.columns:
        kp = np.stack(df[kp_col].to_numpy()).astype(np.float64).reshape(T, 2, 21, 3)
    segments = _blocks(sub_id, text)
    for h, (side, off) in enumerate(_EGO_WRIST.items()):
        pos = st[:, off:off + 16].reshape(T, 4, 4)[:, :3, 3]
        events, dropped, boosted = [], [], []
        if kp is not None:
            from codegen.grounding.aperture import aperture_events, label_boosted_events
            k = kp[:, h]
            events, dropped = aperture_events(k[:, _THUMB_TIP], k[:, _INDEX_TIP],
                                              k[:, _MIDDLE_TIP], fps=fps)
            # annotation-conditioned second pass: recovers small contacts that the
            # episode-relative thresholds miss (see aperture.CONTACT_LABEL)
            events, boosted = label_boosted_events(k[:, _THUMB_TIP], k[:, _INDEX_TIP],
                                                   k[:, _MIDDLE_TIP], events, segments, fps=fps)
        per_arm[side] = {"pos": pos, "grip_events": events, "dropped": dropped,
                         "boosted": boosted}
    return per_arm


def _read(embodiment: str, task: str, episode_index: int, with_video: bool):
    import pandas as pd
    src = source_root(embodiment) / task
    pq, mp4, info = episode_paths(src, episode_index, camera(embodiment))
    if not pq.exists():
        raise FileNotFoundError(f"{pq} (episode {episode_index} of {task})")
    df = pd.read_parquet(pq)
    fps = float(info.get("fps", 30.0))
    clip = None
    if with_video:
        clip = LazyClip(mp4)                     # forward-only lazy frames
        if len(clip) != len(df):                 # trust the shorter; keep alignment
            n = min(len(clip), len(df))
            clip, df = clip[:n], df.iloc[:n]
    instruction = _instruction(src, episode_index, task.replace("_", " "))
    sub_id, text = _sub_tasks(src, df, episode_index, instruction)
    per_arm = _per_arm(embodiment, df, fps, sub_id, text)
    return src, info, fps, clip, instruction, sub_id, text, per_arm


def load_episode(embodiment: str, task: str, root: Path = None,
                 episode_index: int = 0) -> Episode:
    """Load ONE episode (frames + state) of dataset dir ``task`` of a pretraining source."""
    if embodiment not in SOURCES:
        raise ValueError(f"unknown embodiment {embodiment!r} (expected one of {sorted(SOURCES)})")
    src, info, fps, clip, instruction, sub_id, text, per_arm = _read(
        embodiment, task, episode_index, with_video=True)
    return Episode(
        instruction=instruction, clip=clip, sub_id=sub_id, text=text, per_arm=per_arm, fps=fps,
        meta={"instruction": instruction, "T": int(len(sub_id)), "fps": fps,
              "sub_task_text": text, "dataset": task, "episode_index": episode_index},
        episode_id=f"episode_{episode_index:03d}",
    )


# ------------------------------------------------------- STATE-ONLY (no video)
#: The skeleton pipeline's tiers A-inventory and B-assemble never look at a
#: frame: they segment the episode and ground the cached code against the
#: detector, both of which live in the parquet. The contract is the subset of
#: ``Episode`` those tiers consume: {"T", "per_arm", "sub_id", "text", "instruction"},
#: with ``per_arm`` carrying the SAME detector output ``load_episode`` produces.
def load_state(embodiment: str, task: str, root: Path = None,
               episode_index: int = 0) -> dict:
    """State-only load of ONE episode — parquet in, no video opened."""
    if embodiment not in SOURCES:
        raise ValueError(f"unknown embodiment {embodiment!r} (expected one of {sorted(SOURCES)})")
    _src, _info, _fps, _clip, instruction, sub_id, text, per_arm = _read(
        embodiment, task, episode_index, with_video=False)
    return {"T": int(len(sub_id)), "per_arm": per_arm, "sub_id": sub_id,
            "text": text, "instruction": instruction}
