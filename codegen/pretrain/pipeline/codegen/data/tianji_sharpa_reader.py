"""Reader: a Tianji + Sharpa LeRobot labeling view -> pipeline A `Episode`.

Shape of the problem. Pipeline A's two existing readers each supply the whole Episode from one
source: the yam reader derives grip events from an analog gripper channel; the ego reader
derives them from a fingertip-aperture proxy. Sharpa teleop
needs a THIRD combination — a LeRobot dir like yam, but contact events from real per-finger
tactile, and sub-task blocks from `subtasks.json` instead of human annotation.

  clip      <- the view's single 480x640 top-camera stream (a higher resolution than the
               224x320 training view helps the labelling VLM read fine detail)
  per_arm   <- {"pos": wrist xyz from `{side}_wrist_eef`, "grip_events": tactile events}
  sub_id    <- from subtasks.json (tactile splits, VLM-named); see annotate_subtasks_tactile.py
  text      <- ditto
  instruction <- meta/tasks.jsonl

WHY grip_events COME FROM TACTILE. Sharpa has no gripper-width channel to threshold and its hand
kinematics are not in this view, so neither existing detector applies. The tactile detector is a
DIRECT contact sensor (opposition rule: thumb AND (index OR middle)) and is strictly better
grounded than the aperture proxy ego data has to use. Its output is interface-identical to `aperture_events`,
so facts/effects/gate consume it unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HANDS = ("left", "right")


def _read_state(d: Path, episode_index: int):
    import pandas as pd
    info = json.loads((d / "meta" / "info.json").read_text())
    chunk = episode_index // int(info.get("chunks_size", 1000))
    pq = d / info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    mp4 = d / info["video_path"].format(episode_chunk=chunk, episode_index=episode_index,
                                        video_key="top")
    df = pd.read_parquet(pq)
    st = np.stack(df["observation.state"].to_numpy())
    return st, mp4, float(info.get("fps", 30.0)), info


def load_tianji_episode(dataset: str | Path, episode_index: int = 0,
                        subtasks_path: str | Path | None = None):
    from codegen.data.readers import Episode
    from codegen.grounding.tactile import TACTILE_FINGERS, tactile_hand_events

    d = Path(dataset).expanduser().resolve()
    st, mp4, fps, _ = _read_state(d, episode_index)
    mod = json.loads((d / "meta" / "modality.json").read_text())["state"]
    T = len(st)

    tactile = {h: {f: st[:, mod[f"{h}_{f}_f6"]["start"]:mod[f"{h}_{f}_f6"]["end"]]
                   for f in TACTILE_FINGERS if f"{h}_{f}_f6" in mod} for h in HANDS}
    events, _dropped = tactile_hand_events(tactile)

    per_arm = {}
    for h in HANDS:
        s = mod[f"{h}_wrist_eef"]["start"]
        per_arm[h] = {"pos": st[:, s:s + 3].astype(np.float32),   # xyz; rot6d unused downstream
                      "grip_events": list(events.get(h, []))}

    sp = Path(subtasks_path or (d / "subtasks.json"))
    if not sp.exists():
        raise FileNotFoundError(
            f"{sp} missing — run codegen/posttrain/annotate_subtasks_tactile.py first. "
            "Downstream Sharpa data has no human sub-task annotation; that script manufactures "
            "it (tactile splits, VLM names).")
    rec = json.loads(sp.read_text())[str(episode_index)]
    sub_id = np.asarray(rec["sub_id"], dtype=np.int32)
    if len(sub_id) != T:
        raise ValueError(f"subtasks.json ep{episode_index}: sub_id len {len(sub_id)} != T {T} — "
                         "the annotation was built against a different view build")

    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"]
             for l in (d / "meta" / "tasks.jsonl").read_text().splitlines()}
    instruction = rec.get("instruction") or tasks.get(0, "")

    try:
        from codegen.data.frames import LazyClip
        clip = LazyClip(str(mp4))
    except Exception:                       # any decode-backend mismatch -> eager, still correct
        import av
        c = av.open(str(mp4))
        clip = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
        c.close()

    return Episode(instruction=instruction, clip=clip, sub_id=sub_id,
                   text={int(k): v for k, v in rec["text"].items()},
                   per_arm=per_arm, fps=fps,
                   meta={"dataset": str(d), "episode_index": episode_index,
                         "detector": "tactile-opposition", "source": "tianji_sharpa"},
                   episode_id=f"episode_{episode_index:03d}")
