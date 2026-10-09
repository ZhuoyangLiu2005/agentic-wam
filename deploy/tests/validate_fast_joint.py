#!/usr/bin/env python
"""Validate deploy/common/fast_joint_infer.py against the original.

Three phases:
  1. EQUIVALENCE: original vs fast-EAGER (AGENTICWAM_FAST_JOINT_NO_COMPILE) under the
     same torch seed -> the refactor reproduces the exact op order, so actions
     must match to float tolerance and the code string must be identical.
  2. COMPILED: fast-compiled under the same seed -> actions may differ slightly
     (kernel fusion reorders float math; the sampler is stochastic anyway --
     docs/REALWORLD.md §4.5: ~0.05 rad across seeds); report the delta.
  3. TIMING: steady-state original vs fast-compiled.

Run on an IDLE GPU (AGENTICWAM_TEST_CKPT / _DATASET / _INSTRUCTION select the checkpoint; hydra overrides,
e.g. `paths=<name>`, go in AGENTICWAM_TEST_OVERRIDES, space-separated):
    CUDA_VISIBLE_DEVICES=1 $ENV/bin/python deploy/tests/validate_fast_joint.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY))

os.environ.setdefault("AGENTICWAM_DEPLOY_DTYPE", "bf16")

CKPT = os.environ.get("AGENTICWAM_TEST_CKPT", "/path/to/posttrain_yam_checkpoint.pt")
NIS = int(os.environ.get("NIS", "6"))
INSTR = os.environ.get("AGENTICWAM_TEST_INSTRUCTION", "<a task instruction of the checkpoint>")


def seeded_infer(rt, imgs, state, seed=1234):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    chunk, code, ms = rt.infer(imgs, state, INSTR)
    return np.asarray(chunk), code or "", ms


def timed(rt, imgs, state, n=3):
    import torch
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rt.infer(imgs, state, INSTR)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return min(times)


def main() -> int:
    import torch
    from common.policy_runtime import PolicyRuntime
    from common.contract import CAMERAS
    from common import fast_joint_infer

    rt = PolicyRuntime(ckpt=CKPT, task="posttrain_yam",
                       dataset=os.environ.get("AGENTICWAM_TEST_DATASET", "/path/to/yam_dataset"),
                       num_inference_steps=NIS, device="cuda",
                       overrides=os.environ.get("AGENTICWAM_TEST_OVERRIDES", "").split())
    model = rt.policy.model
    rng = np.random.default_rng(0)
    imgs = {c: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for c in CAMERAS}
    state = np.zeros(14)
    failed = False

    for _ in range(2):
        rt.infer(imgs, state, INSTR)

    # 1. equivalence: original vs fast-eager, same seed
    a_orig, c_orig, _ = seeded_infer(rt, imgs, state)
    os.environ["AGENTICWAM_FAST_JOINT_NO_COMPILE"] = "1"
    fast_joint_infer.install(model)
    a_fast, c_fast, _ = seeded_infer(rt, imgs, state)
    d = float(np.abs(a_orig - a_fast).max())
    same_code = c_orig == c_fast
    print(f"1. fast-EAGER vs original: max action diff {d:.2e}, code identical: {same_code}")
    if d > 1e-4 or not same_code:
        print("   FAIL -- the refactor is NOT equivalent")
        failed = True

    # 2. compiled: report drift (expected small; sampler noise scale is ~0.05 rad)
    os.environ.pop("AGENTICWAM_FAST_JOINT_NO_COMPILE")
    model._infer_action_joint = model._infer_action_joint_orig
    fast_joint_infer.install(model)
    rt.infer(imgs, state, INSTR)          # compile + capture
    rt.infer(imgs, state, INSTR)          # warm
    a_comp, c_comp, _ = seeded_infer(rt, imgs, state)
    d2 = float(np.abs(a_orig - a_comp).max())
    print(f"2. fast-COMPILED vs original (same seed): max action diff {d2:.2e} "
          f"(fusion noise; sampler noise is ~5e-2), code identical: {c_orig == c_comp}")
    if d2 > 5e-2:
        print("   FAIL -- compiled drift exceeds sampler noise")
        failed = True

    # 3. timing
    t_comp = timed(rt, imgs, state)
    model._infer_action_joint = model._infer_action_joint_orig
    t_orig = timed(rt, imgs, state)
    print(f"3. timing: original {t_orig:.0f} ms -> fast-compiled {t_comp:.0f} ms "
          f"({100*(t_orig-t_comp)/t_orig:+.0f}%)")
    print("RESULT:", "FAIL" if failed else "ALL OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
