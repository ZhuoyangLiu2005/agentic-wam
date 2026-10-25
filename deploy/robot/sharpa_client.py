"""Client for the deploy server running `--embodiment sharpa` (vega arms + Sharpa hands).

Same transport as `agenticwam_client.py` (one kept-alive HTTP/1.1 socket, Bearer auth,
JPEG/PNG frames, latency stats) -- this subclass only swaps the YAM-specific payload:

                     YAM (agenticwam_client)            Sharpa (this file)
    state            14-D, yam_station schema           58-D [L_arm 7, L_hand 22, R_arm 7, R_hand 22]
    action chunk     (H, 14) abs joints                 (H, 58) abs joints/hand, same layout as state
    cameras          station roles top/left/right       contract keys head_camera/left_camera/right_camera
    frame size       any (server resizes)               exactly 640x360 raw -- the server's
                                                        resolution_guard rejects anything else

Do not resize frames here. The server applies the training crop (deploy/common/sharpa_obs.py),
and that crop is in pixels of the raw 640x360 frame.

The server returns ABSOLUTE targets (it runs IK on the 64-D EEF prediction and adds nothing
relative), so rows are commanded as-is. Sharpa has no RTC: `committed`/`freeze` are never sent.

Observation dict expected by `infer`:

    {"images": {"head_camera": HxWx3 uint8 RGB, "left_camera": ..., "right_camera": ...},
     "state": (58,) float, "capture_ns": int (optional)}
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger
import numpy as np

try:  # on the station both files sit in the same package
    from . import agenticwam_client as base
except ImportError:
    import agenticwam_client as base

STATE_DIM = 58
RAW_SHAPE = (360, 640, 3)
CAMERAS = ("head_camera", "left_camera", "right_camera")
SLICES = {
    "left_arm": slice(0, 7),
    "left_hand": slice(7, 29),
    "right_arm": slice(29, 36),
    "right_hand": slice(36, 58),
}

AgenticWamServerError = base.AgenticWamServerError
AgenticWamClientConfig = base.AgenticWamClientConfig


def print_code(replan: int, code: str | None, previous: str | None) -> None:
    """Print one replan's generated program. With AGENTICWAM_CODE_PIN (the launch_server.sh
    default) the server keeps the step-0 program for the whole episode, so repeats are one line."""
    if not code:
        print(f"[replan {replan}] no code (2-expert checkpoint, or the code expert returned nothing)", flush=True)
    elif code == previous:
        print(f"[replan {replan}] code unchanged ({len(code)} chars)", flush=True)
    else:
        print(f"[replan {replan}] POLICY_CODE:\n{code}\n[/POLICY_CODE]", flush=True)


def check_server_resolution_guard(client: "SharpaPolicyClient", obs: dict, instruction: str) -> str:
    """Camera image size check on the server: send a 320x224 head frame on purpose; the server's
    resolution guard must reject it with HTTP 400 (nothing is inferred). Returns the server's message."""
    import cv2

    small = cv2.resize(obs["images"]["head_camera"], (320, 224), interpolation=cv2.INTER_AREA)
    images = {cam: {"format": "png", "data": base._encode_image(f, "png", 0)}
              for cam, f in {**obs["images"], "head_camera": small}.items()}
    payload = {"protocol": base.PROTOCOL_VERSION, "instruction": instruction, "images": images,
               "state": [float(x) for x in obs["state"]], "seq": -1, "obs_stamp_ns": 0}
    try:
        client._request("POST", "/act", payload)
    except AgenticWamServerError as exc:
        if "HTTP 400" in str(exc):
            return str(exc)
        raise
    raise AgenticWamServerError("server ACCEPTED a 320x224 head frame: its resolution guard is not active")


def split_action_matrix(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """(H, 58) -> {left_arm (H,7), left_hand (H,22), right_arm (H,7), right_hand (H,22)}."""
    return {k: matrix[..., s] for k, s in SLICES.items()}


class SharpaPolicyClient(base.AgenticWamPolicyClient):
    """`AgenticWamPolicyClient` speaking the 58-D vega+Sharpa contract."""

    def connect(self) -> dict[str, Any]:
        """Open the socket, read /health, refuse a server that is not serving Sharpa."""
        logger.info("Connecting to AGENTICWAM deploy server at http://{}:{}", self.config.host, self.config.port)
        self.health, _ = self._request("GET", "/health")
        logger.info(
            "Server ready | task={} embodiment={} action_space={} horizon={} replan={} nis={}",
            self.health.get("task"),
            self.health.get("embodiment"),
            self.health.get("action_space"),
            self.health.get("action_horizon"),
            self.health.get("replan_steps"),
            self.health.get("num_inference_steps"),
        )
        embodiment = self.health.get("embodiment")
        dim = int(self.health.get("action_dim", -1))
        if embodiment != "sharpa" or dim != STATE_DIM:
            raise AgenticWamServerError(
                f"server serves embodiment={embodiment!r} action_dim={dim}; this client speaks 58-D "
                "vega+Sharpa joints. Launch the server with EMBODIMENT=sharpa TASK=posttrain_sharpa."
            )
        if self.health.get("smoke_mode"):
            logger.warning("Server is in SMOKE MODE (no checkpoint): actions are meaningless.")
        if self.config.resize_to is not None:
            raise AgenticWamServerError(
                "resize_to must be None for Sharpa: the server's training crop needs raw 640x360 frames"
            )
        return self.health

    def infer(
        self,
        obs: dict,
        task: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Request one chunk. Returns ((H, 58) absolute targets, info)."""
        images = {}
        for key in CAMERAS:
            frame = obs["images"].get(key)
            if frame is None:
                raise AgenticWamServerError(f"observation is missing the '{key}' camera")
            # Checked here too so a misconfigured camera fails before the frame is sent.
            if frame.shape != RAW_SHAPE or frame.dtype != np.uint8:
                raise AgenticWamServerError(
                    f"{key}: got {frame.shape} {frame.dtype}, need {RAW_SHAPE} uint8 RGB (raw 640x360)"
                )
            images[key] = {
                "format": self.config.image_format,
                "data": base._encode_image(frame, self.config.image_format, self.config.jpeg_quality),
            }

        state = np.asarray(obs["state"], dtype=np.float64).reshape(-1)
        if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
            raise AgenticWamServerError(f"state: need {STATE_DIM} finite values, got {state.shape}")

        payload = {
            "protocol": base.PROTOCOL_VERSION,
            "instruction": task or self.config.instruction,
            "images": images,
            "state": [float(x) for x in state],
            "seq": self._seq,
            "obs_stamp_ns": int(obs.get("capture_ns", time.monotonic_ns())),
        }
        self._seq += 1

        t0 = time.perf_counter()
        try:
            resp, nbytes = self._request("POST", "/act", payload)
        except AgenticWamServerError:
            self.stats.failures += 1
            raise
        round_trip_ms = (time.perf_counter() - t0) * 1000.0

        chunk = resp.get("action_chunk")
        if not isinstance(chunk, list) or not chunk:
            self.stats.failures += 1
            raise AgenticWamServerError("response contained no action_chunk")
        if resp.get("action_type") != base.ACTION_TYPE_ABS:
            self.stats.failures += 1
            raise AgenticWamServerError(
                f"action_type is {resp.get('action_type')!r}, expected {base.ACTION_TYPE_ABS!r}"
            )

        matrix = np.asarray(chunk, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != STATE_DIM:
            raise AgenticWamServerError(f"expected (H, {STATE_DIM}) chunk, got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise AgenticWamServerError("chunk contains NaN or inf")

        timing = resp.get("timing") or {}
        infer_ms = float(timing.get("infer_ms", 0.0))
        queue_ms = float(timing.get("queue_ms", 0.0))
        network_ms = round_trip_ms - infer_ms - queue_ms
        self.stats.record(round_trip_ms, infer_ms, network_ms, nbytes)

        info = {
            "horizon": matrix.shape[0],
            "replan_steps": int(resp.get("replan_steps", matrix.shape[0])),
            "round_trip_ms": round_trip_ms,
            "server_infer_ms": infer_ms,
            "server_queue_ms": queue_ms,
            "network_ms": network_ms,
            "request_bytes": nbytes,
            "code": resp.get("code"),
        }
        return matrix, info

    def __enter__(self) -> SharpaPolicyClient:
        self.connect()
        return self
