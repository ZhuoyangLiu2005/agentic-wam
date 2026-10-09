#!/usr/bin/env python
"""Decisive CUDA-graph test: compile model.mot with reduce-overhead + fullgraph,
log cudagraph skip reasons, and time steady-state inference.

Context (deploy/tests/profile_joint_infer.py findings): eager inference is
LAUNCH-OVERHEAD-bound -- GPU busy only 57% of wall, ~5k tiny kernel launches per
denoise step -- and the MoT forward compiles fullgraph with NO graph breaks.
CUDA graphs are the textbook fix; a plain reduce-overhead attempt on the live
server regressed ~6%, so graphs likely never engaged. TORCH_LOGS=cudagraphs
prints the skip reason if they don't.

Run on an IDLE GPU:
    CUDA_VISIBLE_DEVICES=1 TORCH_LOGS=cudagraphs $ENV/bin/python \
        deploy/tests/profile_cudagraph_test.py
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


def timed(rt, imgs, state, n=3, label=""):
    import torch
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rt.infer(imgs, state, INSTR)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    print(f"{label}: " + " ".join(f"{t:.0f}" for t in times) + " ms")
    return min(times)


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

    for _ in range(2):
        rt.infer(imgs, state, INSTR)
    eager = timed(rt, imgs, state, 3, "eager")

    model = rt.policy.model
    model.mot = torch.compile(model.mot, mode="reduce-overhead", fullgraph=True)
    print("compiling (first call pays capture)...")
    t0 = time.perf_counter()
    rt.infer(imgs, state, INSTR)
    print(f"compile+capture call: {(time.perf_counter()-t0)*1e3:.0f} ms")
    rt.infer(imgs, state, INSTR)   # second call: cudagraph warmup/record
    compiled = timed(rt, imgs, state, 3, "compiled reduce-overhead+fullgraph")
    print(f"\nRESULT: eager {eager:.0f} ms -> compiled {compiled:.0f} ms "
          f"({100*(eager-compiled)/eager:+.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
