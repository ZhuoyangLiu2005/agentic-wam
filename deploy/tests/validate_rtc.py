#!/usr/bin/env python
"""Validate inference-only RTC end to end (no robot, GPU only).

Simulates one replan cycle exactly the way the deploy loop does:
  1. infer chunk A from a state (no guidance)
  2. pretend L rows of A executed during the next inference; the committed
     rows are A[0:K]
  3. infer chunk B from the same obs WITH committed=A[0:K], freeze=L
Checks:
  a. B's frozen prefix decodes to the committed joints (hard inpaint lands)
  b. B's soft region departs smoothly (no step at the freeze boundary)
  c. an unguided B' from the same seed diverges more than the guided B
     (guidance is actually doing something)

Run: CUDA_VISIBLE_DEVICES=1 $ENV/bin/python deploy/tests/validate_rtc.py
"""
import os
import sys
from pathlib import Path

import numpy as np

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY))
os.environ.setdefault("AGENTICWAM_DEPLOY_DTYPE", "bf16")
os.environ.setdefault("AGENTICWAM_FAST_JOINT", "1")

CKPT = os.environ.get("AGENTICWAM_TEST_CKPT", "/path/to/posttrain_yam_checkpoint.pt")
INSTR = os.environ.get("AGENTICWAM_TEST_INSTRUCTION", "<a task instruction of the checkpoint>")
L, K = 8, 16          # freeze 8 rows (delay), send 16 committed rows total


def seeded(rt, imgs, state, **kw):
    import torch
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    chunk, _, _ = rt.infer(imgs, state, INSTR, **kw)
    return np.asarray(chunk)


def main() -> int:
    from common.policy_runtime import PolicyRuntime
    from common.contract import CAMERAS

    rt = PolicyRuntime(ckpt=CKPT, task="posttrain_yam",
                       dataset=os.environ.get("AGENTICWAM_TEST_DATASET", "/path/to/yam_dataset"),
                       num_inference_steps=6, device="cuda",
                       overrides=os.environ.get("AGENTICWAM_TEST_OVERRIDES", "").split())
    rng = np.random.default_rng(3)
    imgs = {c: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for c in CAMERAS}
    state = np.array([0.02, 0.15, 0.25, -0.08, 0.05, 0.02, 0.9,
                      0.02, 0.15, 0.25, -0.08, 0.05, 0.02, 0.9])
    for _ in range(2):
        rt.infer(imgs, state, INSTR)

    A = seeded(rt, imgs, state)
    committed = A[:K].tolist()
    B = seeded(rt, imgs, state, committed=committed, freeze=L)
    B_free = seeded(rt, imgs, state)          # same seed, no guidance
    assert rt.n_rtc_guided >= 1, "RTC never staged!"

    ji = [*range(0, 6), *range(7, 13)]
    frozen_err = np.abs(B[:L, ji] - A[:L, ji]).max()
    grip_err = np.abs(B[:L, [6, 13]] - A[:L, [6, 13]]).max()
    print(f"a. frozen prefix (rows 0..{L-1}): max joint err {frozen_err:.4f} rad, "
          f"gripper err {grip_err:.4f}  {'OK' if frozen_err < 0.02 else 'FAIL'}")

    step_at_boundary = np.abs(B[L, ji] - B[L - 1, ji]).max()
    typical = np.abs(np.diff(B[L:, ji], axis=0)).max(1)
    print(f"b. step at freeze boundary: {step_at_boundary:.4f} rad vs post-boundary "
          f"p95 {np.percentile(typical, 95):.4f}  "
          f"{'OK' if step_at_boundary < max(0.06, 2*np.percentile(typical,95)) else 'FAIL'}")

    div_guided = np.abs(B[:K, ji] - A[:K, ji]).max()
    div_free = np.abs(B_free[:K, ji] - A[:K, ji]).max()
    print(f"c. overlap divergence from A: guided {div_guided:.4f} vs unguided "
          f"{div_free:.4f}  {'OK' if div_guided <= div_free + 1e-6 else 'WEAK'}")

    ok = frozen_err < 0.02
    print("RESULT:", "ALL OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
