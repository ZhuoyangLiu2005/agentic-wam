#!/usr/bin/env python
"""Profile one _infer_action_joint call to find where the ~90 ms/step goes.

Answers, with data:
  1. GPU-busy vs wall time  -> compute-bound or launch-overhead-bound?
  2. Top CUDA ops           -> matmuls (architectural) or elementwise/copies (fixable)?
  3. fullgraph compile      -> the exact graph-break reason torch.compile hits.
  4. Mask-build cost        -> is hoisting _build_mot_attention_mask worth it?

Run on an IDLE GPU (the live server owns GPU 0):
    CUDA_VISIBLE_DEVICES=1 $ENV/bin/python deploy/tests/profile_joint_infer.py
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


def main() -> int:
    import torch
    from common.policy_runtime import PolicyRuntime
    from common.contract import CAMERAS

    rt = PolicyRuntime(ckpt=CKPT, task="posttrain_yam",
                       dataset=os.environ.get("AGENTICWAM_TEST_DATASET", "/path/to/yam_dataset"),
                       num_inference_steps=NIS, device="cuda",
                       overrides=os.environ.get("AGENTICWAM_TEST_OVERRIDES", "").split())
    rng = np.random.default_rng(0)
    imgs = {c: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for c in CAMERAS}
    state = np.zeros(14)

    # warm up twice (CUDA init + code canvas shape)
    for _ in range(2):
        rt.infer(imgs, state, INSTR)

    # 1) wall time, eager
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rt.infer(imgs, state, INSTR)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1e3
    print(f"\n== eager infer wall: {wall_ms:.0f} ms (nis={NIS}) ==")

    # 2) profile
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        rt.infer(imgs, state, INSTR)
    ka = prof.key_averages()
    cuda_total_us = sum(getattr(e, "self_device_time_total", 0) for e in ka)
    print(f"GPU busy total: {cuda_total_us/1e3:.0f} ms of {wall_ms:.0f} ms wall "
          f"({100*cuda_total_us/1e3/wall_ms:.0f}% -> "
          f"{'COMPUTE-bound' if cuda_total_us/1e3/wall_ms > 0.8 else 'LAUNCH/OVERHEAD-bound'})")
    print(ka.table(sort_by="self_device_time_total", row_limit=12))

    # 3) mask-build cost (step-invariant -> hoistable)
    model = rt.policy.model
    dev = rt.device
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10):
        model._build_mot_attention_mask(2520, 32, dev, code_seq_len=550)
    torch.cuda.synchronize()
    print(f"mask build: {(time.perf_counter()-t0)*1e2:.2f} ms/call")

    # 4) fullgraph compile: surface the exact graph-break reason
    try:
        cm = torch.compile(model.mot, fullgraph=True)
        # trigger with a tiny fake call? too invasive; instead compile lazily via one infer
        model_mot_orig, model.mot = model.mot, cm
        try:
            rt.infer(imgs, state, INSTR)
            print("fullgraph compile: OK (no graph breaks)")
            torch.cuda.synchronize(); t0 = time.perf_counter()
            rt.infer(imgs, state, INSTR)
            torch.cuda.synchronize()
            print(f"compiled(fullgraph) infer wall: {(time.perf_counter()-t0)*1e3:.0f} ms")
        finally:
            model.mot = model_mot_orig
    except Exception as e:
        print(f"fullgraph compile FAILED (the graph-break reason):\n{str(e)[:1200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
