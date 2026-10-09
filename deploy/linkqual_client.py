#!/usr/bin/env python
"""Link-qualification CLIENT — runs on the YAM robot host. Stdlib only, no numpy/Pillow.

Measures whether the robot -> GPU link can carry a remote-inference control loop, BEFORE any
control code is written against the wrong latency assumption. Reports:

  1. reachability        does the robot route to the GPU host at all
  2. RTT + jitter        min / p50 / p95 over N pings, with and without connection reuse
                         (the delta IS the cost of deploy_server.py defaulting to HTTP/1.0)
  3. upstream throughput time to ship an obs payload, at both JPEG-sized and PNG-sized bodies
  4. /act simulation     end-to-end round trip at a realistic payload, p50 / p95
  5. timing verdict      does (inference + network) fit inside one chunk of motion

Usage (on the robot host):
  python3 linkqual_client.py --server 10.244.91.75:8712
  python3 linkqual_client.py --server 10.244.91.75:8712 --infer-sec 0.84 --replan 24 --hz 30
"""
import argparse
import http.client
import json
import os
import statistics
import sys
import time

# Realistic observation payloads: 3 cameras, base64 inflates binary by 4/3.
PAYLOADS = [
    ("jpeg q90 @640x480 x3", int(3 * 45_000 * 4 / 3)),
    ("jpeg q90 native x3",   int(3 * 120_000 * 4 / 3)),
    ("png native x3 (today)", int(3 * 500_000 * 4 / 3)),
]


def pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def fmt(ms):
    return f"{ms:7.1f} ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True, help="host:port of linkqual_server.py, e.g. 10.244.91.75:8712")
    ap.add_argument("--pings", type=int, default=30)
    ap.add_argument("--acts", type=int, default=20, help="simulated /act round trips")
    ap.add_argument("--infer-sec", type=float, default=0.84, help="assumed server inference time per /act")
    ap.add_argument("--replan", type=int, default=24, help="rows executed between replans")
    ap.add_argument("--hz", type=float, default=30.0, help="robot control rate")
    ap.add_argument("--timeout", type=float, default=10.0)
    a = ap.parse_args()

    host, _, port = a.server.partition(":")
    port = int(port or 8712)

    def conn():
        return http.client.HTTPConnection(host, port, timeout=a.timeout)

    # ---- 1. reachability -----------------------------------------------------
    print(f"\n== 1. reachability -> {host}:{port} ==")
    t0 = time.perf_counter()
    try:
        c = conn()
        c.request("GET", "/health")
        r = c.getresponse()
        body = json.loads(r.read())
        print(f"   REACHABLE in {(time.perf_counter()-t0)*1e3:.1f} ms | server proto={body.get('proto')}")
    except Exception as e:
        print(f"   NOT REACHABLE: {type(e).__name__}: {e}")
        print("\n   -> robot cannot route to the GPU host. Needs a tunnel or overlay network")
        print("      (WireGuard/Tailscale, SSH reverse tunnel via a bastion, or a dial-out relay).")
        print("      A fast failure usually means 'refused' (routable, nothing listening);")
        print("      a slow timeout usually means blocked/unroutable.")
        return 1

    # ---- 2. RTT, reused vs fresh connection ---------------------------------
    print(f"\n== 2. RTT over {a.pings} pings ==")
    reused = []
    c = conn()
    for _ in range(a.pings):
        t = time.perf_counter()
        c.request("GET", "/ping")
        c.getresponse().read()
        reused.append((time.perf_counter() - t) * 1e3)
    c.close()

    fresh = []
    for _ in range(a.pings):
        t = time.perf_counter()
        c2 = conn()
        c2.request("GET", "/ping")
        c2.getresponse().read()
        c2.close()
        fresh.append((time.perf_counter() - t) * 1e3)

    print(f"   reused conn (keepalive)  min {fmt(min(reused))}  p50 {fmt(pct(reused,50))}  p95 {fmt(pct(reused,95))}")
    print(f"   fresh conn per request   min {fmt(min(fresh))}   p50 {fmt(pct(fresh,50))}  p95 {fmt(pct(fresh,95))}")
    handshake = pct(fresh, 50) - pct(reused, 50)
    print(f"   -> handshake cost {handshake:.1f} ms/request  "
          f"(deploy_server.py pays this today: no protocol_version = HTTP/1.0)")
    jitter = statistics.pstdev(reused) if len(reused) > 1 else 0.0
    print(f"   -> jitter (stdev, reused) {jitter:.1f} ms")

    # ---- 3. upstream throughput at obs-payload sizes ------------------------
    print("\n== 3. upstream (robot -> GPU), obs-sized payloads ==")
    c = conn()
    up_ms = {}
    for label, nbytes in PAYLOADS:
        blob = os.urandom(nbytes)          # incompressible, like real JPEG/PNG bytes
        samples = []
        for _ in range(5):
            t = time.perf_counter()
            c.request("POST", "/echo", body=blob,
                      headers={"Content-Type": "application/octet-stream",
                               "Content-Length": str(len(blob))})
            c.getresponse().read()
            samples.append((time.perf_counter() - t) * 1e3)
        med = pct(samples, 50)
        up_ms[label] = med
        mbps = (nbytes * 8 / 1e6) / (med / 1e3) if med > 0 else 0
        print(f"   {label:<24} {nbytes/1024:7.0f} KiB  p50 {fmt(med)}  ~{mbps:6.1f} Mbps")
    c.close()

    # ---- 4. simulated /act round trip ---------------------------------------
    best_label, best_bytes = PAYLOADS[0]
    print(f"\n== 4. simulated /act round trip ({a.acts}x, payload = {best_label}) ==")
    c = conn()
    acts = []
    blob = os.urandom(best_bytes)
    for _ in range(a.acts):
        t = time.perf_counter()
        c.request("POST", "/echo", body=blob,
                  headers={"Content-Type": "application/octet-stream",
                           "Content-Length": str(len(blob))})
        c.getresponse().read()
        acts.append((time.perf_counter() - t) * 1e3)
    c.close()
    net_p50, net_p95 = pct(acts, 50), pct(acts, 95)
    print(f"   network-only round trip   p50 {fmt(net_p50)}  p95 {fmt(net_p95)}  max {fmt(max(acts))}")

    # ---- 5. timing verdict ---------------------------------------------------
    chunk_ms = a.replan / a.hz * 1e3
    infer_ms = a.infer_sec * 1e3
    total_p50, total_p95 = infer_ms + net_p50, infer_ms + net_p95
    print(f"\n== 5. timing verdict (replan={a.replan} rows @ {a.hz:g} Hz = {chunk_ms:.0f} ms of motion) ==")
    print(f"   inference {infer_ms:.0f} ms + network p50 {net_p50:.0f} ms = {total_p50:.0f} ms  "
          f"-> duty ratio {chunk_ms/total_p50:5.2f}x")
    print(f"   inference {infer_ms:.0f} ms + network p95 {net_p95:.0f} ms = {total_p95:.0f} ms  "
          f"-> duty ratio {chunk_ms/total_p95:5.2f}x")
    if total_p95 <= chunk_ms:
        print("   VERDICT: fits even at p95. A synchronous client keeps up; async pipelining adds margin.")
    elif total_p50 <= chunk_ms:
        print("   VERDICT: fits at p50 but NOT p95 -> the arm stalls on slow requests.")
        print("            Use async pipelining (request the next chunk while executing the current one).")
    else:
        print("   VERDICT: does NOT fit. Async pipelining is mandatory, and you also need one of:")
        print("            fewer denoise steps, a larger replan interval, or a lower control rate.")
        need = chunk_ms - net_p95
        print(f"            At p95 network, inference must drop below {need:.0f} ms to fit this chunk.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
