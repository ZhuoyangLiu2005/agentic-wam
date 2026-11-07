"""Burn every replan's generated program onto real-time (one frame per control step) eval videos.

args: code_log.jsonl run_dir out_dir ckpt_label [episode ids...]
Video frame 0 is the reset observation; frame s (1..800) is after control step s. Policy call k (1-based) sees the
frame at step 24(k-1) and its chunk runs steps 24(k-1)+1 .. 24k, so call k's program is shown from frame 24(k-1).
The k-th logged episode with >= 3 calls is eval episode k (single env, in order; the server's extra reset logs none).
Writes <out_dir>/<ckpt_label>_ep<NN>.mp4 and <out_dir>/<ckpt_label>_index.json (episode -> outcome, calls, file).
"""
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict

here = os.path.dirname(os.path.abspath(__file__))
log, run_dir, out_dir, label = sys.argv[1:5]
want = {int(x) for x in sys.argv[5:]}
REPLAN = 24
STRIDE = int(os.environ.get("OVERLAY_STRIDE", REPLAN))   # video frames per policy call: 24 real-time, 1 per-call snapshots
FPS = float(os.environ.get("OVERLAY_FPS", 25))
FIRST = int(os.environ.get("OVERLAY_FIRST_EP", 0))
os.makedirs(out_dir, exist_ok=True)

by_ep = defaultdict(list)
for line in open(log):
    r = json.loads(line)
    by_ep[r["episode"]].append(r)
eps = [rows for _, rows in sorted(by_ep.items()) if len(rows) >= 3]
res = json.load(open(os.path.join(run_dir, "_result.json"))).get("details", {})
index = {}
for k, rows in enumerate(eps):
    eid = FIRST + k
    if want and eid not in want:
        continue
    vids = glob.glob(os.path.join(run_dir, f"episode_{eid:07d}_cam_head_*.mp4"))
    if not vids:
        continue
    rows.sort(key=lambda r: r["chunk"])
    n = len(rows)
    segs = [{"start": STRIDE * (r["chunk"] - 1), "code": r["code"] or "",
             "counter": f"replan {r['chunk']}/{n}  ·  step {REPLAN * (r['chunk'] - 1)}",
             "label": f"call {r['chunk']} runs steps {REPLAN * (r['chunk'] - 1) + 1}–{REPLAN * r['chunk']}"}
            for r in rows]
    det = res.get(str(eid), {})
    outcome = "success" if det.get("success") else "fail"
    distinct = len({r["code"] for r in rows})
    spec = {"title": f"{label} · eval episode {eid}",
            "subtitle": f"{outcome}, score {det.get('score', '?')} · {n} policy calls · {distinct} distinct programs · "
                        + ("real time, 25 fps" if STRIDE == REPLAN else f"one frame per policy call, {FPS:g} fps"),
            "fps": FPS, "flash_frames": 8 if STRIDE == REPLAN else 1, "segments": segs}
    sp = os.path.join(out_dir, f"{label}_ep{eid:02d}.json")
    json.dump(spec, open(sp, "w"), indent=1)
    out = os.path.join(out_dir, f"{label}_ep{eid:02d}.mp4")
    subprocess.run([sys.executable, "-I", os.path.join(here, "code_overlay.py"), vids[0], sp, out], check=True)
    index[eid] = {"outcome": outcome, "score": det.get("score"), "calls": n, "distinct": distinct,
                  "file": os.path.basename(out), "codes": [r["code"] or "" for r in rows]}
json.dump(index, open(os.path.join(out_dir, f"{label}_index.json"), "w"), indent=1)
print(f"{label}: {len(index)} episodes -> {out_dir}")
