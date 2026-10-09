#!/usr/bin/env python
"""REMOTE inference server — model on the H100 cluster, robot is a thin GPU-less client.

Line B of the deploy infra. Line A (`deploy/deploy_server.py`) stays as-is for the GPU-on-robot
case; both drive the same `deploy/common/policy_runtime.py`, so the motion-producing path is
shared and cannot drift.

What this adds over Line A, all of it because the link is now a real network:

  keepalive     protocol_version = "HTTP/1.1". Line A leaves it at the HTTP/1.0 default, so every
                /act pays a fresh TCP handshake -- invisible on loopback, a full RTT over a WAN.
  auth          Bearer token, constant-time compare. This endpoint MOVES A ROBOT ARM; on a shared
                cluster it must not be open. Binding a non-loopback address without a token is
                refused unless you pass --insecure explicitly.
  backpressure  Inference is serialized (one GPU). A second in-flight request waits up to
                --max-wait-sec, then gets 503 instead of piling up unboundedly behind the first.
  telemetry     Every response carries {queue_ms, infer_ms, server_ms}. The client subtracts these
                from its round trip to get the true network cost -- no guessing where time went.
  loud errors   Full tracebacks to the log, always. A swallowed exception here once looked exactly
                like a model result and nearly became a published false negative
                (reports/FINAL_code_drift_experiments.md §6).
  warmup        First inference is 3-5x slower (CUDA init); it is burned at startup, not on a live
                observation with the arm powered.

Usage:
  export AGENTICWAM_DEPLOY_TOKEN=$(python3 -c "import secrets;print(secrets.token_urlsafe(24))")
  python deploy/remote/server.py \
      --ckpt "runs/<run>/checkpoints/weights/step_400000.pt" \
      --task posttrain_yam --dataset <dataset_dir> \
      --num-inference-steps 10 --replan-steps 24 --port 8712
"""
import argparse
import hmac
import json
import logging
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# deploy/remote/server.py -> parents[1] == deploy/, so `common` is importable as a package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.contract import (ActRequest, ActResponse, ContractError,  # noqa: E402
                             PROTOCOL_VERSION, ACTION_TYPE_ABS)
from common import obs_codec  # noqa: E402

LOOPBACK = ("127.0.0.1", "localhost", "::1")
log = logging.getLogger("agenticwam.deploy.remote")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained step_XXXXXX.pt (nonexistent path -> smoke mode)")
    ap.add_argument("--task", default="posttrain_yam", help="the training task config (configs/real_post_train/task/)")
    ap.add_argument("--dataset", default=None, help="the dataset directory the checkpoint was post-trained on (its stats + processing)")
    ap.add_argument("--override", action="append", default=[], help="extra hydra override, e.g. model.infer_code_tokens=192 (repeatable)")
    ap.add_argument("--dataset-stats", default=None, help="default: cfg.data.train.pretrained_norm_stats")
    ap.add_argument("--action-horizon", type=int, default=None, help="default: num_frames-1 (=32)")
    ap.add_argument("--replan-steps", type=int, default=24,
                    help="advertised to the client; >=26 needed at ~0.84s inference and 30Hz")
    ap.add_argument("--num-inference-steps", type=int, default=10, help="denoise steps; main speed/quality knob")
    ap.add_argument("--instruction", default="put the plastic bottles in the bin", help="fallback instruction")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8712)
    ap.add_argument("--token", default=None, help="shared secret; default: $AGENTICWAM_DEPLOY_TOKEN")
    ap.add_argument("--insecure", action="store_true", help="allow a non-loopback bind with NO auth")
    ap.add_argument("--max-body-mb", type=float, default=32.0, help="reject larger request bodies")
    ap.add_argument("--max-wait-sec", type=float, default=5.0, help="queue wait before 503")
    ap.add_argument("--socket-timeout", type=float, default=30.0, help="per-connection socket timeout")
    ap.add_argument("--no-warmup", action="store_true", help="skip the startup warmup inference")
    ap.add_argument("--video-log-dir", default=None, help="save the model's imagined-future video")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    token = a.token or os.environ.get("AGENTICWAM_DEPLOY_TOKEN")
    if not token and a.host not in LOOPBACK and not a.insecure:
        log.error("refusing to bind %s with no token: this endpoint commands a robot arm.\n"
                  "  set AGENTICWAM_DEPLOY_TOKEN=<secret>, or pass --insecure if the link is already "
                  "protected (e.g. an SSH tunnel bound to loopback).", a.host)
        return 2
    if not token:
        log.warning("running WITHOUT authentication (host=%s)", a.host)

    from common.policy_runtime import PolicyRuntime   # imported late: pulls in torch

    rt = PolicyRuntime(ckpt=a.ckpt, task=a.task, dataset=a.dataset, overrides=a.override, dataset_stats=a.dataset_stats,
                       action_horizon=a.action_horizon, replan_steps=a.replan_steps,
                       num_inference_steps=a.num_inference_steps, device=a.device,
                       video_log_dir=a.video_log_dir)

    if rt.smoke_mode:
        log.warning("SMOKE MODE: checkpoint missing, actions are meaningless. Do NOT drive a robot.")
    if not a.no_warmup:
        log.info("warming up (first inference pays CUDA init)...")
        log.info("warmup done in %.2fs", rt.warmup())

    gate = threading.BoundedSemaphore(1)   # one GPU -> one inference at a time
    max_body = int(a.max_body_mb * 1024 * 1024)
    stats = {"n_act": 0, "n_err": 0, "n_503": 0}

    def _authorized(handler) -> bool:
        if not token:
            return True
        got = handler.headers.get("Authorization", "")
        if got.startswith("Bearer "):
            got = got[len("Bearer "):]
        return hmac.compare_digest(got, token)

    def do_act(body, recv_ns):
        req = ActRequest.from_dict(body, expect_state_dim=rt.action_dim)
        instruction = req.instruction or a.instruction
        images = obs_codec.decode_observation(req.images)

        t_queue = time.perf_counter()
        if not gate.acquire(timeout=a.max_wait_sec):
            stats["n_503"] += 1
            raise TimeoutError(f"inference busy for >{a.max_wait_sec}s (another request in flight)")
        try:
            queue_ms = (time.perf_counter() - t_queue) * 1e3
            # Optional RTC fields (inference-only real-time chunking): the
            # client may send the joint actions it has already committed to
            # executing (`committed`, [K,14]) plus how many of them cover the
            # inference delay (`freeze`). ActRequest ignores unknown keys, so
            # this stays wire-compatible with older clients.
            chunk, code, infer_ms = rt.infer(images, req.state, instruction,
                                             committed=body.get("committed"),
                                             freeze=body.get("freeze"))
        finally:
            gate.release()

        send_ns = time.time_ns()
        resp = ActResponse(
            action_chunk=chunk.tolist(), horizon=int(chunk.shape[0]), action_dim=int(chunk.shape[1]),
            replan_steps=rt.replan_steps, action_type=ACTION_TYPE_ABS, instruction=instruction,
            code=code, seq=req.seq, obs_stamp_ns=req.obs_stamp_ns,
            timing={"queue_ms": round(queue_ms, 2), "infer_ms": round(infer_ms, 2),
                    "server_ms": round((send_ns - recv_ns) / 1e6, 2)},
        )
        stats["n_act"] += 1
        log.info("/act seq=%d %s | queue %.0fms infer %.0fms server %.0fms%s",
                 req.seq, obs_codec.describe(req.images), queue_ms, infer_ms,
                 (send_ns - recv_ns) / 1e6, "" if code is None else f" | code {len(code)}B")
        return resp.to_dict()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"          # keepalive: no handshake per /act
        server_version = "agenticwam-deploy-remote/1"

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass                                # we do our own structured logging

        def _read_body(self):
            n = int(self.headers.get("Content-Length", 0))
            if n > max_body:
                raise ContractError(f"body of {n/1e6:.1f} MB exceeds --max-body-mb={a.max_body_mb}")
            remaining, buf = n, []
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                buf.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(buf)
            return json.loads(raw) if raw else {}

        def do_GET(self):
            if self.path == "/ping":            # unauthenticated liveness only
                return self._send(200, {"ok": True, "protocol": PROTOCOL_VERSION})
            if not _authorized(self):
                return self._send(401, {"error": "unauthorized: send 'Authorization: Bearer <token>'"})
            if self.path == "/health":
                info = dict(rt.info())
                info.update(ok=True, protocol=PROTOCOL_VERSION, auth=bool(token), **stats)
                return self._send(200, info)
            self._send(404, {"error": "GET /ping or /health"})

        def do_POST(self):
            recv_ns = time.time_ns()
            if not _authorized(self):
                return self._send(401, {"error": "unauthorized: send 'Authorization: Bearer <token>'"})
            try:
                body = self._read_body()
            except ContractError as e:
                return self._send(413, {"error": str(e)})
            except Exception as e:
                return self._send(400, {"error": f"bad request body: {e}"})

            try:
                if self.path == "/reset":
                    ep = rt.reset()
                    log.info("/reset -> episode %d", ep)
                    self._send(200, {"ok": True, "reset": True, "episode": ep})
                elif self.path == "/act":
                    self._send(200, do_act(body, recv_ns))
                else:
                    self._send(404, {"error": "POST /act or /reset"})
            except ContractError as e:
                stats["n_err"] += 1
                log.warning("contract error on %s: %s", self.path, e)
                self._send(400, {"error": f"contract: {e}"})
            except TimeoutError as e:
                log.warning("busy: %s", e)
                self._send(503, {"error": str(e), "retry": True})
            except Exception as e:
                stats["n_err"] += 1
                log.error("UNHANDLED on %s: %s\n%s", self.path, e, traceback.format_exc())
                self._send(500, {"error": str(e), "trace": traceback.format_exc()[-1200:]})

    srv = ThreadingHTTPServer((a.host, a.port), H)
    srv.timeout = a.socket_timeout
    log.info("READY | task=%s code_expert=%s dtype=%s | horizon=%d replan=%d nis=%d | auth=%s",
             rt.task, rt.use_code, rt.model_dtype, rt.action_horizon, rt.replan_steps,
             rt.num_inference_steps, bool(token))
    log.info("listening on http://%s:%d  (POST /act /reset, GET /health /ping)", a.host, a.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down | served %d /act, %d errors, %d busy-rejects",
                 stats["n_act"], stats["n_err"], stats["n_503"])
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
