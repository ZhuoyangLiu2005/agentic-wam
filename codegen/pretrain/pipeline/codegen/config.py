"""Workspace configuration — the single home for every USER-SETTABLE setting.
Precedence: CLI FLAG (per-invocation override) > ENV VARIABLE > the config
file named by the CODEGEN_CONFIG env var (path relative to the workspace
root — e.g. `CODEGEN_CONFIG=configs/ego.yaml`; unset = no config file) >
built-in default. With a filled config file the launch commands stay bare
(`$PY scripts/run_batch.py --tasks-file <units>`); CLI flags exist only to
override a setting for one invocation.

Dataset paths have NO built-in default on purpose: they are deployment state,
not code. Start from the templates `configs/{yam,ego}.yaml` (set the dataset
root, launch with `CODEGEN_CONFIG=configs/<file>.yaml`); a missing path raises
immediately with instructions.

Keys (this run — dataset / version / mode):
    EMBODIMENT        which embodiment this process annotates (yam | ego, or the
                      post-training yam_sft | tianji_sharpa); the CODEGEN_EMBODIMENT
                      env var wins over the config chain (see ``embodiment()``)
    TAG               campaign version tag (run dirs, batch dirs, frontier
                      scoping) — set once per campaign
    VANILLA           1/true = annotation-only prompt AND uniform sampling
Keys (datasets — docs/PRETRAINING.md §1):
    YAM_ROOT          directory of YAM LeRobot datasets (train_shard_* …)
    EGO_ROOT          directory of ego LeRobot datasets
    SELECTION         optional subset of work units to label: a text file of
                      `<dataset>[:<episode>]` lines, or a JSON list of
                      {"subset": <dataset>, "episode_index": i}. Unset = every
                      episode of every dataset that ships the camera stream
    YAM_CAMERA        video stream the actor sees (default cam_high)
    EGO_CAMERA        same for ego
    EGO_KEYPOINTS     optional per-frame fingertip keypoints column of ego data
                      (float[2*21*3]); absent = no contact events for ego
    NOISE_LABELS      comma-separated sub-task labels that mark annotator noise
                      (skipped, exact normalized match); default none
Keys (actor):
    VLM_URL           OpenAI-compatible chat-completions endpoint of the actor VLM
                      (key: env VLM_API_KEY or a .vlm_api_key file)
    CODER_MODEL       actor VLM (model id at that endpoint)
    GRIP_MIN_DWELL    frames: events closer than this form a burst; a burst that
                      is also STATIONARY is collapsed to its net transition
                      (0 = off, detector output unchanged)
    GRIP_HOVER_EPS    metres of TCP travel below which a burst counts as hover
    YAM_FRAME_GAP     max frames between consecutive VLM frames of a yam slice
    EGO_FRAME_GAP     same for ego (denser: human hands move faster)
    MAX_SLICE_FRAMES  hard ceiling on frames per VLM call (0 = unlimited) —
                      keeps a long slice from exceeding the router's body limit
    REPAIR_TAIL       1 = insert gripper events the coder dropped at a slice
                      boundary, from the sensor (marked `[auto-repair]`, counted
                      as tail_repaired; raw and repaired rates both reported)
    REPAIR_RETRIES    max repair rounds per slice
    MAX_SLICES        cap on coded slices per episode (0 = all; debug knob)
    VLM_MAX_TOKENS    per-call output cap for the actor
    VLM_TEMPERATURE   actor sampling temperature
    SKELETON_STORE    skeleton cache dir (default outputs/<emb>/skeleton/store)
Keys (batch / coordinator):
    PARALLEL          concurrent actors per batch
    TASKS_PER_ROUND   tasks per coordinator round (dataset partitioning)
    ACTOR_TIMEOUT     per-episode actor subprocess timeout (seconds)
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_DEFAULTS = {                       # pipeline defaults (deployment paths have NONE)
    "EMBODIMENT": "yam",
    "TAG": "",                      # campaigns must set one (config file or --tag)
    "VANILLA": "0",
    "MAX_SLICES": "0",
    "CODER_MODEL": "gcp/google/gemini-3.1-pro-preview",
    "TASKS_PER_ROUND": "8",
    "PARALLEL": "8",
    # per-episode actor subprocess wall clock (a hung VLM call otherwise occupies
    # a worker slot invisibly); timeouts are recorded as re-queue candidates
    "ACTOR_TIMEOUT": "7200",
    "YAM_CAMERA": "observation.images.cam_high",
    "EGO_CAMERA": "observation.images.cam_high",
    "EGO_KEYPOINTS": "observation.hand_keypoints",
    "NOISE_LABELS": "",
    # Frame sampling is GAP-based (slices differ in length): endpoints + event
    # pre-frames are anchors, and any interval longer than the gap is filled
    # with evenly spaced frames (no events -> uniform sampling). Human hands move
    # fast -> 20 frames (~0.67 s at 30 fps); the robot is slower -> 40 (~1.33 s).
    "YAM_FRAME_GAP": "40",
    "EGO_FRAME_GAP": "20",
    # HARD ceiling on frames per VLM call (0 = unlimited): the router rejects
    # very large request bodies, so a long slice is capped. Fillers are dropped
    # first; endpoints and event pre-frames last.
    "MAX_SLICE_FRAMES": "48",
    # hover-chatter suppression on the YAM gripper detector (transitions.py).
    # 0 = OFF and byte-identical to the raw detector.
    "GRIP_MIN_DWELL": "0",
    "GRIP_HOVER_EPS": "0.02",
    # deterministic tail repair: when the coder under-emits gripper events at a
    # slice boundary, insert the missing ones from the sensor (marked, counted).
    "REPAIR_TAIL": "1",
    # skeleton-cached labelling (codegen.skeleton): where the per-label code
    # cache lives. Derived state — regenerable from the inventory + the model.
    "SKELETON_STORE": "",
    "REPAIR_RETRIES": "2",
    "VLM_MAX_TOKENS": "4096",
    "VLM_TEMPERATURE": "0.1",
}


def _file_cfg() -> dict:
    # CODEGEN_CONFIG names THE config file (path relative to the workspace
    # root) — e.g. per-embodiment files under configs/ so two concurrent
    # coordinator sessions each read their own settings. Unset = no config
    # file: env vars + built-in defaults only.
    sel = os.environ.get("CODEGEN_CONFIG")
    if not sel:
        return {}
    p = Path(sel) if os.path.isabs(sel) else ROOT / sel
    if not p.exists():
        raise RuntimeError(f"CODEGEN_CONFIG={sel!r} does not exist ({p})")
    import yaml
    data = yaml.safe_load(p.read_text()) or {}
    # An EMPTY yaml value parses to None; str(None) would be the literal "None",
    # which is TRUTHY and would sail through every `if not value` guard (a blank
    # `tag:` then writes run dirs named "None"). Empty means unset.
    return {str(k).upper(): ("" if v is None else str(v)) for k, v in data.items()}


_FILE = _file_cfg()


def get(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key) or _FILE.get(key) or _DEFAULTS.get(key, default)


def get_int(key: str) -> int:
    return int(get(key))


def get_float(key: str) -> float:
    return float(get(key))


def get_bool(key: str) -> bool:
    return str(get(key) or "").strip().lower() in ("1", "true", "yes", "on")


def embodiment() -> str:
    """The active embodiment: legacy CODEGEN_EMBODIMENT env var first (every
    doc and dispatch line uses it), then the EMBODIMENT config chain."""
    return os.environ.get("CODEGEN_EMBODIMENT") or get("EMBODIMENT")


def get_path(key: str) -> Path:
    v = get(key)
    if not v:
        raise RuntimeError(
            f"{key} is not configured — set the env var or add it to "
            f"your config file (CODEGEN_CONFIG=configs/<file>.yaml — start "
            f"from configs/yam.yaml or configs/ego.yaml)")
    return Path(v)


def frame_gap(embodiment: str) -> int:
    """Max frames between consecutive VLM frames of a slice for this embodiment."""
    return get_int("EGO_FRAME_GAP" if embodiment == "ego" else "YAM_FRAME_GAP")


def video_stream(embodiment: str) -> str:
    """The camera stream the actor sees — part of every generation cache key."""
    if embodiment == "yam":
        return get("YAM_CAMERA")
    if embodiment == "ego":
        return get("EGO_CAMERA")
    return "video_main"
