#!/usr/bin/env python3
"""Replay a recorded T-Rex episode through the Sharpa deploy server -- NOTHING MOVES.

Milestone 1 of the vega+Sharpa client (no robot driver yet). It runs the same loop a live
run will:

    /health -> /reset -> [observe at t -> POST /act -> "execute" replan_steps rows -> t += rows]

but the observation at t is the RECORDED frame + 58-D state, and executing a row just compares it
with the recorded action at that step. So it checks the whole client <-> server path (contract,
camera keys, raw 640x360 frames, 58-D state in, (H, 58) absolute chunk out, latency) and reports
open-loop action error against the demonstration.

Reads the T-Rex gateway LeRobot v3.0 dataset, NOT the converted v2.1 shard: that shard's videos
are already cropped to 224x320, and the server's resolution guard only accepts the raw 640x360
frames the training crop is defined on.

    uv run --no-project --with av --with loguru --with pandas --with pyarrow --with opencv-python-headless \\
        python deploy/robot/run_sharpa_replay.py --token-file /tmp/agenticwam_token.txt

The server (launched with EMBODIMENT=sharpa TASK=posttrain_sharpa) is reached at 127.0.0.1:8712
by default -- forward the port from the GPU host.
"""

from __future__ import annotations

import argparse
import datetime as dt
from fractions import Fraction
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sharpa_client  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO / "data_local/trex_gateway_tong_transfer_sf_norawtac_20260820"
FPS = 30

# contract camera key -> T-Rex v3.0 video key (same mapping as utils/convert_trex_v30_to_v21.py)
VIDEO_KEYS = {
    "head_camera": "observation.images.head_left",
    "left_camera": "observation.images.left_wrist",
    "right_camera": "observation.images.right_wrist",
}


class FrameStream:
    """Sequential decode of one episode's frames from a packed v3.0 mp4.

    Mirrors T-Rex's unpack_lerobot_to_raw_fk.decode_frames (seek, then index by PTS) without
    importing it -- that module pulls in pinocchio/h5py, which the client side does not need.
    Indices passed to `get` must not decrease.
    """

    def __init__(self, root: Path, ep: pd.Series, key: str):
        import av

        prefix = f"videos/{key}/"
        path = (root / "videos" / key / f"chunk-{int(ep[prefix + 'chunk_index']):03d}"
                / f"file-{int(ep[prefix + 'file_index']):03d}.mp4")
        self.start = int(round(float(ep[prefix + "from_timestamp"]) * FPS))
        self._container = av.open(str(path))
        self._stream = self._container.streams.video[0]
        if self.start:
            self._container.seek(int(Fraction(self.start, FPS) / self._stream.time_base),
                                 stream=self._stream, backward=True, any_frame=False)
        self._frames = self._container.decode(self._stream)
        self._pos = -1
        self._frame = None

    def get(self, index: int) -> np.ndarray:
        """RGB frame at episode-relative `index`."""
        if index < self._pos:
            raise ValueError(f"FrameStream is forward-only: asked {index} after {self._pos}")
        while self._pos < index:
            frame = next(self._frames)
            pos = round(float(frame.pts * self._stream.time_base) * FPS) - self.start
            if pos < 0:
                continue
            if pos != self._pos + 1:
                raise RuntimeError(f"video gap: expected frame {self._pos + 1}, got {pos}")
            self._pos, self._frame = pos, frame
        return self._frame.to_ndarray(format="rgb24")

    def close(self) -> None:
        self._container.close()


class ReplayEpisode:
    """One recorded episode, exposed the way a live robot driver would be."""

    def __init__(self, root: Path, episode: int):
        eps = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))])
        rows = eps[eps.episode_index == episode]
        if rows.empty:
            raise SystemExit(f"episode {episode} not in {root}")
        ep = rows.iloc[0]
        self.length = int(ep.length)

        data = (root / "data" / f"chunk-{int(ep['data/chunk_index']):03d}"
                / f"file-{int(ep['data/file_index']):03d}.parquet")
        d = pd.read_parquet(data, filters=[("episode_index", "=", episode)]).sort_values("frame_index")
        if len(d) != self.length:
            raise RuntimeError(f"episode {episode}: {len(d)} rows, expected {self.length}")
        self.state = np.stack(d["observation.state"].to_numpy()).astype(np.float64)
        self.action = np.stack(d["action"].to_numpy()).astype(np.float64)

        self._streams = {cam: FrameStream(root, ep, key) for cam, key in VIDEO_KEYS.items()}

    def get_observation(self, t: int) -> dict:
        return {"images": {cam: s.get(t) for cam, s in self._streams.items()},
                "state": self.state[t], "capture_ns": int(t * 1e9 / FPS)}

    def close(self) -> None:
        for s in self._streams.values():
            s.close()


def group_error(pred: np.ndarray, ref: np.ndarray) -> dict[str, dict[str, float]]:
    """Mean/max |pred - ref| per body part (arms in rad, hands in the dataset's hand units)."""
    out = {}
    for name, s in sharpa_client.SLICES.items():
        e = np.abs(pred[..., s] - ref[..., s])
        out[name] = {"mean": float(e.mean()), "max": float(e.max())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="T-Rex LeRobot v3.0 root")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--instruction", default=None, help="default: the dataset's single task string")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8712)
    ap.add_argument("--token-file", type=Path, default=None, help="default: $AGENTICWAM_DEPLOY_TOKEN")
    ap.add_argument("--image-format", choices=["jpeg", "png"], default="jpeg")
    ap.add_argument("--max-replans", type=int, default=None, help="stop early (default: whole episode)")
    ap.add_argument("--guard-test", action="store_true", help="also check the server rejects a resized frame")
    ap.add_argument("--out", type=Path, default=None, help="report JSON (default: runs/sharpa_replay/...)")
    a = ap.parse_args()

    token = a.token_file.read_text().strip() if a.token_file else os.environ.get("AGENTICWAM_DEPLOY_TOKEN", "")
    root = a.dataset.resolve()
    instruction = a.instruction or str(pd.read_parquet(root / "meta/tasks.parquet").index[0])

    episode = ReplayEpisode(root, a.episode)
    client = sharpa_client.SharpaPolicyClient(sharpa_client.AgenticWamClientConfig(
        host=a.host, port=a.port, token=token, instruction=instruction, image_format=a.image_format))

    records, t, prev_tail, prev_code = [], 0, None, None
    try:
        health = client.connect()
        client.reset(task=instruction)
        print(f"episode {a.episode}: {episode.length} frames | instruction: {instruction[:80]}...")

        while t < episode.length - 1:
            if a.max_replans is not None and len(records) >= a.max_replans:
                break
            obs = episode.get_observation(t)
            if a.guard_test and not records:
                msg = sharpa_client.check_server_resolution_guard(client, obs, instruction)
                print(f"guard test OK: server rejected a 320x224 head frame ({msg})")
            chunk, info = client.infer(obs, task=instruction)
            rows = min(info["replan_steps"], len(chunk), episode.length - t)

            rec = {
                "replan": len(records), "t": t, "rows": rows,
                "round_trip_ms": round(info["round_trip_ms"], 1),
                "server_infer_ms": round(info["server_infer_ms"], 1),
                "network_ms": round(info["network_ms"], 1),
                # open-loop error of the executed rows vs the demonstration
                "action_error": group_error(chunk[:rows], episode.action[t:t + rows]),
                # how far row 0 sits from the state it was conditioned on
                "row0_vs_state": group_error(chunk[0], obs["state"]),
                # jump from the previous chunk's last executed row to this chunk's row 0
                "seam_jump": None if prev_tail is None else group_error(chunk[0], prev_tail),
                "code": info["code"],
            }
            records.append(rec)
            err = rec["action_error"]
            print(f"replan {rec['replan']:3d} t={t:5d} | rtt {rec['round_trip_ms']:6.0f} ms "
                  f"(model {rec['server_infer_ms']:5.0f}) | err mean L/R arm "
                  f"{err['left_arm']['mean']:.3f}/{err['right_arm']['mean']:.3f} "
                  f"hand {err['left_hand']['mean']:.3f}/{err['right_hand']['mean']:.3f}")
            sharpa_client.print_code(rec["replan"], info["code"], prev_code)
            prev_tail, prev_code = chunk[rows - 1], info["code"]
            t += rows
    finally:
        client.close()
        episode.close()

    if not records:
        print("no replans ran")
        return 1

    def agg(field: str) -> dict:
        vals = [r[field] for r in records if r[field] is not None]
        return {g: {"mean": float(np.mean([v[g]["mean"] for v in vals])),
                    "max": float(np.max([v[g]["max"] for v in vals]))} for g in sharpa_client.SLICES} if vals else {}

    report = {
        "episode": a.episode, "dataset": str(root), "instruction": instruction,
        "server": {k: health.get(k) for k in ("task", "ckpt", "embodiment", "action_space", "action_horizon",
                                              "replan_steps", "num_inference_steps", "smoke_mode")},
        "frames_covered": t, "episode_length": episode.length, "n_replans": len(records),
        "latency": client.stats.summary(),
        "action_error": agg("action_error"), "row0_vs_state": agg("row0_vs_state"), "seam_jump": agg("seam_jump"),
        "replans": records,
    }
    out = a.out or REPO / "runs/sharpa_replay" / f"ep{a.episode}_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    lat = report["latency"]["round_trip_ms"]
    print(f"\n{len(records)} replans over {t}/{episode.length} frames | rtt p50 {lat['p50']:.0f} ms "
          f"p95 {lat['p95']:.0f} ms")
    for field in ("action_error", "row0_vs_state", "seam_jump"):
        print(f"{field:14s} " + "  ".join(f"{g} {v['mean']:.3f}/{v['max']:.3f}" for g, v in report[field].items()))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
