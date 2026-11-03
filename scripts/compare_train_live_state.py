#!/usr/bin/env python
"""Compare the robot STATE the model was trained on with the state it sees LIVE -- read-only, NOTHING MOVES.

Training: raw 76-D `observation.state` of the v2.1 dataset -> 64-D unified EEF (TianjiSharpaUnifiedEEFTransform,
absolute EEF pos3+rot6d, 22-D hand, zero gripper per arm): the vector the model is conditioned on.
Live: `steps.jsonl` of a run_sharpa_eval.py episode (`state` = 58-D vega joints + hands). The arms go through
FK (deploy/common/vega_eef64.state58_to_raw76, needs pinocchio, i.e. the deploy venv) to the same 64-D space.
Without pinocchio only the two 22-D hands are compared (they are identical in 58-D and 76-D).

    ~/venvs/agenticwam-deploy/bin/python scripts/compare_train_live_state.py \\
        --dataset data_local/trex_tong_transfer_aw76_v21 \\
        --live runs/sharpa_eval/<session>/ep_000/steps.jsonl

Two comparisons, per group of dims (L eef pos / L rot6d / L hand / R ...):
  1. START POSE: live first state vs the first state of every training episode (z-score against that spread).
  2. WHOLE EPISODE: live replan-start states (rows with row==0) vs dataset_stats_eef64.json (global q01..q99, z).
Note: `steps.jsonl` logs robot.get_state() right after each row is commanded, so it is within one control step
(33 ms) of what the model saw.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "deploy/common"))
from sharpa_heldout_infer import Unified64, load_episode  # noqa: E402

# 64-D layout: L eef 0:9 (pos 0:3, rot6d 3:9), L hand 9:31, L grip 31 | R eef 32:41, R hand 41:63, R grip 63
GROUPS = {"L pos": slice(0, 3), "L rot6d": slice(3, 9), "L hand": slice(9, 31),
          "R pos": slice(32, 35), "R rot6d": slice(35, 41), "R hand": slice(41, 63)}
HAND_ONLY = ("L hand", "R hand")


def live_states(steps_jsonl: Path, replan_start_only: bool) -> np.ndarray:
    rows = [json.loads(l) for l in steps_jsonl.read_text().splitlines() if l.strip()]
    if replan_start_only:
        rows = [r for r in rows if r.get("row") == 0]
    s = np.array([r["state"] for r in rows], np.float64)
    if s.ndim != 2 or s.shape[1] != 58:
        raise SystemExit(f"expected [N,58] live states in {steps_jsonl}, got {s.shape}")
    return s


def live_to_64(state58: np.ndarray, u: Unified64):
    """-> ([N,64], hands_only). FK needs pinocchio; without it the hand dims are rebuilt directly."""
    try:
        import vega_eef64
        raw76 = np.stack([vega_eef64.state58_to_raw76(s) for s in state58])
        return u.abs64(raw76), False
    except Exception as e:  # ImportError (pinocchio) or URDF load failure
        print(f"!! FK unavailable ({type(e).__name__}: {e}); comparing the hands only\n")
        out = np.full((len(state58), 64), np.nan)
        out[:, 9:31], out[:, 41:63] = state58[:, 7:29], state58[:, 36:58]
        return out, True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=REPO / "data_local/trex_tong_transfer_aw76_v21")
    ap.add_argument("--live", type=Path, required=True, help="ep_NNN/steps.jsonl from run_sharpa_eval.py")
    ap.add_argument("--start-episodes", type=int, nargs="+", default=list(range(0, 100, 5)),
                    help="training episodes whose first state gives the start-pose spread")
    ap.add_argument("--z-warn", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=REPO / "runs/compare_obs")
    a = ap.parse_args()

    dataset = a.dataset.resolve()
    stats = json.loads((dataset / "dataset_stats_eef64.json").read_text())["state"]["default"]
    g_mean, g_std = np.array(stats["global_mean"]), np.array(stats["global_std"]).clip(min=1e-6)
    q01, q99 = np.array(stats["global_q01"]), np.array(stats["global_q99"])

    u = Unified64()
    live_all, hands_only = live_to_64(live_states(a.live, False), u)
    live_rp, _ = live_to_64(live_states(a.live, True), u)
    groups = {k: v for k, v in GROUPS.items() if not hands_only or k in HAND_ONLY}

    shard = dataset / "train_shard_0"
    starts = np.stack([u.abs64(load_episode(shard, e)["state"][:1])[0] for e in a.start_episodes])

    issues = 0
    print(f"live: {a.live} ({len(live_all)} states, {len(live_rp)} replan starts)")
    print(f"\n1. START POSE vs first state of {len(starts)} training episodes (z = (live - train mean) / train std)")
    s0 = live_all[0]
    mu, sd = starts.mean(0), starts.std(0).clip(min=1e-3)
    for name, sl in groups.items():
        z = np.abs((s0[sl] - mu[sl]) / sd[sl])
        bad = int((z > a.z_warn).sum())
        issues += bad > 0
        print(f"  {name:8s} max |z| {z.max():6.1f}  mean |z| {z.mean():5.1f}  dims > {a.z_warn:g}: {bad}/{z.size}"
              f"{'  <--' if bad else ''}")

    print(f"\n2. WHOLE EPISODE vs dataset_stats_eef64.json (replan-start states)")
    for name, sl in groups.items():
        x = live_rp[:, sl]
        z = np.abs((x - g_mean[sl]) / g_std[sl])
        outside = float(((x < q01[sl]) | (x > q99[sl])).mean())
        bad = bool(z.max() > 5 or outside > 0.10)
        issues += bad
        print(f"  {name:8s} max |z| {z.max():6.1f}  outside [q01,q99] {outside * 100:5.1f}% of values"
              f"{'  <--' if bad else ''}")

    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / "compare_state.npz", live64=live_all, live64_replan_starts=live_rp, train_starts=starts)
    print("\nResult:", f"{issues} group(s) flagged (<--): live state is unlike training"
          if issues else "live state is inside the training distribution")
    print("Arm groups are skipped without FK." if hands_only else "")
    print(f"saved {a.out / 'compare_state.npz'}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
