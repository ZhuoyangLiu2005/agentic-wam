#!/usr/bin/env python
"""Link-qualification SERVER — runs on the GPU host, no model, no GPU.

Answers the question the real deploy server can't answer until it's too late: can the robot
reach this machine, and if so, is the link fast enough to close a 30 Hz control loop through it?

Pair with deploy/linkqual_client.py on the robot host. Endpoints:
  GET  /health           -> {ok, server_ns}                 reachability
  GET  /ping             -> {server_ns}                     RTT (tiny body both ways)
  POST /echo             -> {n_bytes, server_ns}            upstream throughput (obs-sized bodies)
  GET  /blob?bytes=N     -> N bytes                         downstream throughput

Unlike deploy_server.py this sets protocol_version = "HTTP/1.1", so connections are kept alive
and a reused socket costs no extra handshake. The client measures both ways, which quantifies
exactly what the one-line keepalive fix is worth on YOUR link.

Usage:
  python deploy/linkqual_server.py --port 8712
"""
import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

MAX_BLOB = 64 * 1024 * 1024

# Files the robot host may pull down over this link, so bootstrapping needs no working SSH/scp.
# Explicit allowlist -- this must never become a general file server on a shared cluster.
SERVABLE = {"linkqual_client.py", "client.py", "contract.py", "obs_codec.py"}
DEPLOY = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8712)
    a = ap.parse_args()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"   # keepalive; deploy_server.py still defaults to HTTP/1.0

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/health":
                self._send(200, {"ok": True, "server_ns": time.time_ns(),
                                 "pid": os.getpid(), "proto": self.request_version})
            elif u.path == "/ping":
                self._send(200, {"server_ns": time.time_ns()})
            elif u.path == "/blob":
                n = int((parse_qs(u.query).get("bytes") or ["1048576"])[0])
                n = max(0, min(n, MAX_BLOB))
                self._send(200, b"\0" * n, ctype="application/octet-stream")
            elif u.path == "/file":
                name = os.path.basename((parse_qs(u.query).get("name") or ["linkqual_client.py"])[0])
                if name not in SERVABLE:
                    return self._send(403, {"error": f"not servable: {name}", "allowed": sorted(SERVABLE)})
                hits = [p for p in (DEPLOY / name, DEPLOY / "remote" / name, DEPLOY / "common" / name)
                        if p.is_file()]
                if not hits:
                    return self._send(404, {"error": f"{name} not found under {DEPLOY}"})
                self._send(200, hits[0].read_bytes(), ctype="text/plain; charset=utf-8")
            else:
                self._send(404, {"error": "GET /health /ping /blob?bytes=N /file?name=..."})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            remaining, got = n, 0
            while remaining > 0:                       # drain the body fully before replying
                chunk = self.rfile.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                got += len(chunk)
                remaining -= len(chunk)
            self._send(200, {"n_bytes": got, "server_ns": time.time_ns()})

    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"[linkqual] listening on http://{a.host}:{a.port}  "
          f"(GET /health /ping /blob?bytes=N, POST /echo)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
