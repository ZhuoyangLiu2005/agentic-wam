#!/usr/bin/env python3
"""Measure the AGENTICWAM-Cosmos deploy link and decide what control rate it can sustain.

Speaks the `deploy/common/contract.py` protocol against `deploy/remote/server.py`.
Needs no robot and no cameras -- it synthesizes camera-shaped frames -- so it can
run while the station's data-collection loop still owns the hardware.

The question it answers is not "is the link fast" but "does the timing close":

    A chunk row i is the plan for obs_time + i. When the chunk lands, L =
    latency x control_hz rows are already in the past. Only (horizon - L) rows
    are still usable, and they must cover the L rows the *next* request will
    consume. So closed-loop control needs

        horizon - L >= L      i.e.    horizon >= 2 x L

    Below that the queue drains faster than it refills and the arm stalls,
    however the client is tuned.

    python scripts/measure_agenticwam_link.py --token-file /tmp/agenticwam_token.txt --requests 12
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import statistics
import time
import http.client

import cv2
import numpy as np

CAMERAS = ("head_camera", "left_camera", "right_camera")


def synth_frame(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Camera-like content: gradient plus texture.

    Content matters only for JPEG size -- flat colour would understate the
    payload, pure noise would overstate it.
    """
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            (xx / width * 255).astype(np.uint8),
            (yy / height * 255).astype(np.uint8),
            ((xx + yy) / (width + height) * 255).astype(np.uint8),
        ],
        axis=-1,
    )
    noise = rng.integers(-24, 24, size=base.shape, dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def encode(frame: np.ndarray, fmt: str, quality: int) -> str:
    ext = ".jpg" if fmt == "jpeg" else ".png"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if fmt == "jpeg" else []
    ok, buf = cv2.imencode(ext, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), params)
    if not ok:
        raise RuntimeError(f"{fmt} encode failed")
    return base64.b64encode(buf.tobytes()).decode()


class Session:
    """One kept-alive HTTP/1.1 connection, reused for every request.

    This matters more than it looks. `urllib.request` opens a fresh TCP
    connection per call; over a tunnel with a 270 ms RTT that is an extra
    handshake on every single /act. The deploy server sets HTTP/1.1 specifically
    so a client can hold one socket open -- a client that doesn't is measuring
    its own overhead, not the link.
    """

    def __init__(self, host: str, port: int, token: str, timeout: float):
        self.host, self.port, self.token, self.timeout = host, port, token, timeout
        self._conn: http.client.HTTPConnection | None = None
        self.reconnects = 0

    def _connect(self) -> http.client.HTTPConnection:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            self._conn.connect()
        return self._conn

    def request(self, method: str, path: str, payload: dict | None = None) -> tuple[dict, int]:
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Authorization": f"Bearer {self.token}", "Connection": "keep-alive"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        for attempt in (0, 1):
            try:
                conn = self._connect()
                conn.request(method, path, body=body, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {data[:200].decode(errors='replace')}")
                return json.loads(data.decode()), len(body or b"")
            except (http.client.HTTPException, OSError):
                # Server closed an idle socket, or the tunnel blipped. Rebuild
                # once, then give up -- a second failure is not a stale socket.
                self._conn = None
                self.reconnects += 1
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8712)
    ap.add_argument("--token-file", default="/tmp/agenticwam_token.txt")
    ap.add_argument("--requests", type=int, default=12)
    ap.add_argument("--format", choices=("jpeg", "png"), default="jpeg")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--control-hz", type=float, default=30.0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--instruction", default="put the plastic bottles in the bin")
    args = ap.parse_args()

    token = pathlib.Path(args.token_file).read_text().strip()
    sess = Session(args.host, args.port, token, args.timeout)

    health, _ = sess.request("GET", "/health")
    print("=== /health ===")
    for key in ("ok", "task", "code_expert", "dtype", "action_horizon", "replan_steps",
                "n_cameras", "action_dim", "num_inference_steps"):
        if key in health:
            print(f"  {key:20s} {health[key]}")
    horizon = int(health.get("action_horizon", 32))

    # /ping isolates the network floor from the payload and the model.
    pings = []
    for _ in range(10):
        t0 = time.perf_counter()
        try:
            sess.request("GET", "/ping")
            pings.append((time.perf_counter() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001
            print(f"  ping failed: {exc}")
    if pings:
        print(f"\n=== /ping (no payload) ===\n  p50 {pct(pings,50):.1f} ms   p95 {pct(pings,95):.1f} ms")

    sess.request("POST", "/reset", {})

    rng = np.random.default_rng(0)
    images = {
        cam: {"format": args.format, "data": encode(synth_frame(args.width, args.height, rng),
                                                    args.format, args.quality)}
        for cam in CAMERAS
    }
    state = [0.0] * 14

    print(f"\n=== /act x{args.requests} ({args.format}, {args.width}x{args.height}) ===")
    rtts, infer_ms, queue_ms, sizes = [], [], [], []
    for i in range(args.requests):
        payload = {
            "protocol": 1,
            "instruction": args.instruction,
            "images": images,
            "state": state,
            "seq": i,
            "obs_stamp_ns": time.monotonic_ns(),
        }
        t0 = time.perf_counter()
        try:
            resp, nbytes = sess.request("POST", "/act", payload)
        except Exception as exc:  # noqa: BLE001
            print(f"  request {i}: FAILED {exc}")
            continue
        rtt = (time.perf_counter() - t0) * 1000
        timing = resp.get("timing") or {}
        rtts.append(rtt)
        infer_ms.append(float(timing.get("infer_ms", 0.0)))
        queue_ms.append(float(timing.get("queue_ms", 0.0)))
        sizes.append(nbytes)
        tag = "  (warmup)" if i == 0 else ""
        print(f"  {i:2d}  rtt {rtt:7.0f} ms   infer {timing.get('infer_ms', 0):7.0f}   "
              f"rows {len(resp['action_chunk'])}{tag}")

    if not rtts:
        print("\nNo successful /act calls -- cannot assess the link.")
        raise SystemExit(1)

    # Drop the first call: CUDA/allocator warmup makes it 3-5x slower and it is
    # not representative of the steady state the control loop lives in.
    steady = rtts[1:] or rtts
    steady_infer = infer_ms[1:] or infer_ms
    net = [r - i - q for r, i, q in zip(steady, steady_infer, queue_ms[1:] or queue_ms, strict=False)]

    print("\n=== steady state (first call dropped) ===")
    print(f"  round trip   p50 {pct(steady,50):7.0f} ms   p95 {pct(steady,95):7.0f} ms   max {max(steady):7.0f} ms")
    print(f"  server infer p50 {pct(steady_infer,50):7.0f} ms")
    print(f"  network      p50 {pct(net,50):7.0f} ms   (round trip - infer - queue)")
    print(f"  payload      {statistics.fmean(sizes)/1024:.0f} KB/request")
    print(f"  reconnects   {sess.reconnects}  (socket rebuilds during the run)")

    # --- the timing budget -------------------------------------------------
    lat_s = pct(steady, 95) / 1000.0
    hz = args.control_hz
    L = lat_s * hz
    usable = horizon - L
    print("\n=== timing budget ===")
    print(f"  horizon           {horizon} rows = {horizon/hz:.2f} s of motion @ {hz:.0f} Hz")
    print(f"  latency (p95)     {lat_s:.2f} s = {L:.1f} control steps")
    print(f"  usable rows       {usable:.1f}  (horizon - latency)")
    print(f"  need              >= {L:.1f}  (must cover the next request)")

    if usable >= L:
        print(f"\n  CLOSES at {hz:.0f} Hz. Suggested client settings:")
        print(f"    --queue.target-depth {int(np.ceil(L))+2}  --queue.exec-horizon {horizon}")
    else:
        max_hz = (horizon / (2.0 * lat_s)) if lat_s > 0 else 0.0
        print(f"\n  DOES NOT CLOSE at {hz:.0f} Hz -- the queue drains faster than it refills.")
        print(f"  Max sustainable rate with this horizon/latency: {max_hz:.1f} Hz")
        print(f"  Or: cut latency below {horizon/(2.0*hz):.2f} s to hold {hz:.0f} Hz.")
    print()


if __name__ == "__main__":
    main()
