#!/usr/bin/env python
"""Offline validation of deploy/common/yam_eef64.py against the recorded sample trajectory.

Three checks, no GPU and no model -- pure kinematics against real recorded data
(utils/yam_fk_ik/sample_trajectory, an actual robot episode):

  1. state14_to_82 -> the 9-D EEF blocks must match the exported state_64d.npy.
  2. Per-step replay (REPLAY_GUIDE.md loop) must reproduce ik_replay_joints_14d.npy.
  3. Chunk-anchored decode: build a delta chunk from the recorded action poses
     exactly the way YamUnifiedEEFTransform._action64 does (all rows anchored to
     the chunk-start state), run decode_chunk, and compare the recovered joints
     to the recorded absolute joint targets. This is the code path /act uses.

Run on any node with the agenticwam env:
    $ENV/bin/python deploy/tests/validate_yam_eef64.py
"""
import sys
from pathlib import Path

import numpy as np

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY))
from common import yam_eef64 as bridge          # noqa: E402
import yam_kinematics as yk                     # noqa: E402  (path set by bridge)

SAMPLE = Path(yk.__file__).resolve().parent / "sample_trajectory"
CHUNK = 32


def main() -> int:
    s14 = np.load(SAMPLE / "raw_state_14d.npy")
    a14 = np.load(SAMPLE / "raw_action_14d.npy")
    s64 = np.load(SAMPLE / "state_64d.npy")
    d64 = np.load(SAMPLE / "action_delta_64d.npy")
    replay = np.load(SAMPLE / "ik_replay_joints_14d.npy")
    T = len(s14)
    print(f"sample: T={T} steps")
    failed = False

    # 1. state14_to_82 -> EEF must match the exported unified state
    errs = []
    for t in range(0, T, max(1, T // 200)):
        r82 = bridge.state14_to_82(s14[t])
        ML, MR = r82[0:16].reshape(4, 4), r82[16:32].reshape(4, 4)
        eef = np.concatenate([
            ML[:3, 3], ML[:3, 0], ML[:3, 1],
            MR[:3, 3], MR[:3, 0], MR[:3, 1],
        ])
        ref = np.concatenate([s64[t, 0:9], s64[t, 32:41]])
        errs.append(np.abs(eef - ref).max())
    e1 = max(errs)
    print(f"1. state14_to_82 vs state_64d      max err {e1:.2e}  {'OK' if e1 < 1e-4 else 'FAIL'}")
    failed |= e1 >= 1e-4

    # 2. per-step decode, re-anchored at the RECORDED state each step -- what
    #    row 0 of every served chunk experiences (the server re-anchors at the
    #    measured robot state every replan; nothing is ever chained open-loop
    #    across chunks). Compare against the recorded absolute joint targets.
    #    Calibration: the exporter's own round trip is mean 0.46 mrad /
    #    p95 2.5 mrad (sample_trajectory/meta.json). NOTE a fully CHAINED
    #    2579-step rollout is NOT a valid check: one different IK branch at a
    #    near-singular elbow puts the whole remainder on another solution arm.
    errs2 = []
    for t in range(0, T, max(1, T // 400)):
        q = s14[t].copy()
        for arm, js, je, gi, off in [("left", 0, 6, 6, 0), ("right", 7, 13, 13, 32)]:
            T_cur = yk.fk(q[js:je], arm)
            ok, qj = yk.ik(yk.apply_delta_9d(T_cur, d64[t, off:off + 9]), arm, init6=q[js:je])
            q[js:je] = qj
            q[gi] = d64[t, off + 31]
        errs2.append(np.abs(q - a14[t]).max())
    e2p, e2m = np.percentile(errs2, 95), max(errs2)
    print(f"2. per-step decode vs recorded     p95 {e2p:.2e} max {e2m:.2e}  "
          f"{'OK' if e2p < 5e-3 else 'FAIL'}")
    failed |= e2p >= 5e-3

    # 3. chunk-anchored decode (the /act path): deltas of the next CHUNK action
    #    poses w.r.t. the CURRENT state pose, all rows one anchor -- then IK back.
    errs, ikf = [], 0
    for t0 in range(0, T - CHUNK, max(1, (T - CHUNK) // 20)):
        base = s14[t0]
        TbL, TbR = yk.fk_both(base[0:6], base[7:13])
        chunk = np.zeros((CHUNK, 64))
        for k in range(CHUNK):
            row = a14[t0 + k]
            TaL, TaR = yk.fk_both(row[0:6], row[7:13])
            chunk[k, 0:9] = yk.eef_delta_9d(TbL, TaL)
            chunk[k, 31] = row[6]
            chunk[k, 32:41] = yk.eef_delta_9d(TbR, TaR)
            chunk[k, 63] = row[13]
        dec, fails = bridge.decode_chunk(chunk, base)
        ikf += fails
        ref = a14[t0:t0 + CHUNK]
        errs.append(np.abs(dec[:, [*range(0, 6), *range(7, 13)]]
                           - ref[:, [*range(0, 6), *range(7, 13)]]).max())
        g = np.abs(dec[:, [6, 13]] - ref[:, [6, 13]]).max()
        errs.append(g)
    # Tolerance 2e-2 rad: the IK stops at 1 mm / 1 mrad task-space tolerance, so
    # worst-case joint-space error of a few mrad is expected (meta.json: the
    # exporter's own max exceeds its p95 of 2.5 mrad). 20 mrad is still well
    # under the station's 50 mrad/step safety clamp and the ~50 mrad sampler noise.
    e3 = max(errs)
    print(f"3. chunk decode vs recorded joints max err {e3:.2e}  (ik failures {ikf})  "
          f"{'OK' if e3 < 2e-2 else 'FAIL'}")
    failed |= e3 >= 2e-2

    print("RESULT:", "FAIL" if failed else "ALL OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
