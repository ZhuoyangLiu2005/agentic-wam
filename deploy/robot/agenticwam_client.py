"""Client for the AGENTICWAM-Cosmos deploy server (`deploy/remote/server.py`).

Speaks the HTTP contract in `agenticwam_cosmos/deploy/common/contract.py`. Drops into
`AsyncChunkQueue` as its `infer_fn`, so the chunk alignment, cross-fade blending
and safety envelope are unchanged from the ZeroMQ path -- only the transport and
the observation mapping differ.

**Hold one socket open.** This is the single most important line in the file.
`urllib` opens a fresh TCP connection per request; measured through a
port-forward that cost more than the payload did:

    /ping p50   274 ms -> 49 ms          network per /act  770 ms -> 185 ms
    /ping p95  1364 ms -> 88 ms          round trip p95   2337 ms -> 1085 ms

The server sets HTTP/1.1 for exactly this reason. A client that reconnects per
call is measuring its own handshakes, not the link.

Mapping to the station
----------------------
Cameras -- the server wants three, the station has four; `bottom` is unused:

    head_camera  <- top     (cam_high, top-down)
    left_camera  <- left    (cam_left_wrist)
    right_camera <- right   (cam_right_wrist)

State/action layout is `[Ljoint0-5, Lgrip, Rjoint0-5, Rgrip]` -- byte-identical
to `contract.schema.ACTION_KEYS` order, so no reordering is needed. Gripper units
are 0..1 on both sides (verified against the training norm stats: gripper dims
span 0.0002..1.0037). Joints are radians and the dataset's range sits inside the
station's own joint limits.

The server returns **absolute** qpos (`action_type: "qpos_abs"`) -- it has already
added the request's state back to the model's relative prediction. Command the
values directly; do not add state again.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import http.client
import json
import statistics
import time
from typing import Any

from loguru import logger
import numpy as np

# Camera role on the station -> camera key the policy expects.
CAMERA_MAP = {"top": "head_camera", "left": "left_camera", "right": "right_camera"}

PROTOCOL_VERSION = 1
ACTION_TYPE_ABS = "qpos_abs"
ACTION_DIM = 14


class AgenticWamServerError(RuntimeError):
    """The deploy server returned an error, or could not be reached."""


def _encode_image(frame: np.ndarray, fmt: str, quality: int) -> str:
    """JPEG/PNG-encode an RGB frame to base64.

    JPEG q90 is ~10-30x smaller than PNG for the same frame. It is lossy, which
    is a real if small domain shift -- `deploy/common/obs_codec.py` notes the
    model trained on H.264-decoded frames, so it has seen artifacts of a similar
    character. `png` stays available as the reference path.
    """
    import cv2

    ext = ".jpg" if fmt == "jpeg" else ".png"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if fmt == "jpeg" else []
    ok, buf = cv2.imencode(ext, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), params)
    if not ok:
        raise AgenticWamServerError(f"{fmt} encode failed")
    return base64.b64encode(buf.tobytes()).decode()


@dataclass
class AgenticWamClientConfig:
    host: str = "127.0.0.1"
    """With an SSH tunnel / port forward this stays 127.0.0.1."""

    port: int = 8712
    token: str = ""
    """Bearer token. The server refuses a non-loopback bind without one."""

    instruction: str = "put the plastic bottles in the bin"
    image_format: str = "jpeg"
    jpeg_quality: int = 90

    resize_to: tuple[int, int] | None = None
    """Optional client-side downscale, (width, height).

    Default None: send native and let the server apply the training-identical
    `ResizeSmallestSideAspectPreserving -> CenterCrop(320x224)`. Pre-shrinking
    saves bandwidth but any mismatch with that transform silently breaks
    pixel-parity with training. Measured, it is not worth it -- the network is
    ~18% of the round trip and the model is the rest.
    """

    request_timeout_s: float = 30.0
    """Generous. A 3-expert forward is ~840 ms and the tail is longer."""

    connect_timeout_s: float = 15.0


@dataclass
class AgenticWamLatencyStats:
    round_trip_ms: list[float] = field(default_factory=list)
    infer_ms: list[float] = field(default_factory=list)
    network_ms: list[float] = field(default_factory=list)
    request_bytes: list[int] = field(default_factory=list)
    failures: int = 0
    reconnects: int = 0
    _max: int = 200

    def record(self, rt: float, infer: float, net: float, nbytes: int) -> None:
        for series, value in (
            (self.round_trip_ms, rt),
            (self.infer_ms, infer),
            (self.network_ms, net),
            (self.request_bytes, nbytes),
        ):
            series.append(value)
            if len(series) > self._max:
                del series[0]

    @staticmethod
    def _s(series: list[float]) -> dict[str, float]:
        if not series:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        o = sorted(series)
        return {
            "mean": statistics.fmean(series),
            "p50": o[len(o) // 2],
            "p95": o[min(len(o) - 1, int(0.95 * len(o)))],
            "max": o[-1],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "round_trip_ms": self._s(self.round_trip_ms),
            "server_infer_ms": self._s(self.infer_ms),
            "network_ms": self._s(self.network_ms),
            "request_kb": {k: v / 1024 for k, v in self._s([float(b) for b in self.request_bytes]).items()},
            "failures": self.failures,
            "reconnects": self.reconnects,
            "samples": len(self.round_trip_ms),
        }


class AgenticWamPolicyClient:
    """One kept-alive HTTP/1.1 connection to the deploy server."""

    def __init__(self, config: AgenticWamClientConfig | None = None):
        self.config = config or AgenticWamClientConfig()
        self.stats = AgenticWamLatencyStats()
        self._conn: http.client.HTTPConnection | None = None
        self._seq = 0
        self.health: dict[str, Any] = {}

    # --- transport -----------------------------------------------------------

    def _connect(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(
                self.config.host, self.config.port, timeout=self.config.request_timeout_s
            )
            self._conn.connect()
        return self._conn

    def _request(self, method: str, path: str, payload: dict | None = None) -> tuple[dict, int]:
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Authorization": f"Bearer {self.config.token}", "Connection": "keep-alive"}
        if body is not None:
            headers["Content-Type"] = "application/json"

        for attempt in (0, 1):
            try:
                conn = self._connect()
                conn.request(method, path, body=body, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                if resp.status == 503:
                    raise AgenticWamServerError("server busy (503): another request is still inferring")
                if resp.status != 200:
                    raise AgenticWamServerError(
                        f"HTTP {resp.status}: {data[:200].decode(errors='replace')}"
                    )
                return json.loads(data.decode()), len(body or b"")
            except (http.client.HTTPException, OSError) as exc:
                # An idle keepalive socket closed by the server, or a tunnel
                # blip. Rebuild once. A second failure is not a stale socket.
                self._conn = None
                self.stats.reconnects += 1
                if attempt == 1:
                    raise AgenticWamServerError(f"transport failure: {exc}") from exc
        raise AgenticWamServerError("unreachable")

    def connect(self) -> dict[str, Any]:
        """Open the socket and read /health. Returns the server's config."""
        logger.info("Connecting to AGENTICWAM deploy server at http://{}:{}", self.config.host, self.config.port)
        self.health, _ = self._request("GET", "/health")
        logger.info(
            "Server ready | task={} code_expert={} dtype={} horizon={} nis={}",
            self.health.get("task"),
            self.health.get("code_expert"),
            self.health.get("dtype"),
            self.health.get("action_horizon"),
            self.health.get("num_inference_steps"),
        )
        dim = int(self.health.get("action_dim", ACTION_DIM))
        if dim != ACTION_DIM:
            raise AgenticWamServerError(
                f"server serves action_dim={dim}; this client speaks 14-D YAM joints. "
                "A 64-D YAM checkpoint is served as 14-D joints (FK/IK on the server, "
                "docs/REALWORLD.md §4.4); any other action_dim is a different robot."
            )
        return self.health

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # --- endpoints -----------------------------------------------------------

    def ping(self) -> float:
        t0 = time.perf_counter()
        self._request("GET", "/ping")
        return (time.perf_counter() - t0) * 1000.0

    def reset(self, task: str | None = None) -> None:
        """Start a new episode. Clears the server's per-episode code cache."""
        self._seq = 0
        self._request("POST", "/reset", {})
        if task:
            self.config.instruction = task

    def infer(
        self,
        obs: dict,
        task: str | None = None,
        step: int = 0,  # noqa: ARG002 -- the server tracks its own episode step
        previous_action: dict[str, np.ndarray] | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Request an action chunk. Signature matches `AsyncChunkQueue.infer_fn`.

        `previous_action` is accepted and ignored: this server's contract has no
        inpainting field, so chunk seams are handled entirely by the client's
        cross-fade.
        """
        from yam_station.contract import schema

        images = {}
        for role, key in CAMERA_MAP.items():
            frame = obs["images"].get(role)
            if frame is None:
                raise AgenticWamServerError(f"observation is missing the '{role}' camera")
            if self.config.resize_to is not None:
                import cv2

                frame = cv2.resize(frame, self.config.resize_to, interpolation=cv2.INTER_AREA)
            images[key] = {
                "format": self.config.image_format,
                "data": _encode_image(frame, self.config.image_format, self.config.jpeg_quality),
            }

        state = schema.flatten_state(obs)  # (14,) in [Ljoint, Lgrip, Rjoint, Rgrip] order

        payload = {
            "protocol": PROTOCOL_VERSION,
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

        # A relative chunk commanded as absolute would drive the arm to the
        # wrong place, so this is checked, not assumed.
        if resp.get("action_type") != ACTION_TYPE_ABS:
            self.stats.failures += 1
            raise AgenticWamServerError(
                f"action_type is {resp.get('action_type')!r}, expected {ACTION_TYPE_ABS!r}"
            )

        matrix = np.asarray(chunk, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != ACTION_DIM:
            raise AgenticWamServerError(f"expected (H, {ACTION_DIM}) chunk, got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise AgenticWamServerError("chunk contains NaN or inf")

        timing = resp.get("timing") or {}
        infer_ms = float(timing.get("infer_ms", 0.0))
        queue_ms = float(timing.get("queue_ms", 0.0))
        network_ms = round_trip_ms - infer_ms - queue_ms
        self.stats.record(round_trip_ms, infer_ms, network_ms, nbytes)

        info = {
            "horizon": matrix.shape[0],
            "round_trip_ms": round_trip_ms,
            "server_infer_ms": infer_ms,
            "server_queue_ms": queue_ms,
            "network_ms": network_ms,
            "request_bytes": nbytes,
            "code": resp.get("code"),
        }
        return schema.split_action_matrix(matrix), info

    def __enter__(self) -> AgenticWamPolicyClient:
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def timing_budget(latency_s: float, horizon: int, control_hz: float) -> dict[str, Any]:
    """Does the loop close at this rate, and if not, what rate does it close at?

    A chunk row i is the plan for obs_time + i. When the chunk lands, L =
    latency x control_hz rows are already in the past. The remaining
    (horizon - L) rows must cover the L rows the *next* request will consume:

        horizon - L >= L    <=>    control_hz <= horizon / (2 x latency)

    Below that the queue drains faster than it refills and no amount of client
    tuning helps -- the horizon is simply too short for the latency.
    """
    steps = latency_s * control_hz
    usable = horizon - steps
    closes = usable >= steps
    return {
        "latency_s": latency_s,
        "control_hz": control_hz,
        "horizon": horizon,
        "latency_steps": steps,
        "usable_rows": usable,
        "closes": closes,
        "max_control_hz": horizon / (2.0 * latency_s) if latency_s > 0 else float("inf"),
        "target_depth": int(np.ceil(steps)) + 2,
        "exec_horizon": horizon,
    }
