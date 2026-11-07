"""Shared settings for the robodojo-pipeline check scripts, from the environment.

RD_VIEW        training view dir (train_shard_0/, dataset_stats_eef64.json, splits)   required
RD_SRC         RoboDojo LeRobot v2.1 source root (default <repo>/RoboDojo/data/RoboDojo_lerobot_v21_video)
RD_TASK_INDEX  source task_index (default: parsed from the train_shard_0 symlink target train_shard_<k>)
RD_TRAIN_EPS   JSON list of trained episodes (default <RD_VIEW>/episodes_0_94.json)
"""
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
_v = os.environ.get("RD_VIEW")
if not _v:
    raise SystemExit("set RD_VIEW to the training view dir (the one with train_shard_0/)")
VIEW = Path(_v).resolve()
SHARD = VIEW / "train_shard_0"
SRC = Path(os.environ.get("RD_SRC", REPO / "RoboDojo/data/RoboDojo_lerobot_v21_video"))
if os.environ.get("RD_TASK_INDEX"):
    TASK = int(os.environ["RD_TASK_INDEX"])
else:
    m = re.search(r"train_shard_(\d+)$", os.path.realpath(SHARD))
    if not m:
        raise SystemExit(f"cannot infer task index from {os.path.realpath(SHARD)}; set RD_TASK_INDEX")
    TASK = int(m.group(1))
TRAIN_EPS = Path(os.environ.get("RD_TRAIN_EPS", VIEW / "episodes_0_94.json"))
