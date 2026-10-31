#!/usr/bin/env python3
"""Replay recorded T-Rex episodes through the running Sharpa deploy server and plot predicted vs recorded actions.

Same loop as deploy/robot/run_sharpa_replay.py (recorded raw frames + 58-D state -> POST /act -> compare the
executed rows with the demonstration), but it KEEPS the predicted rows so they can be plotted. Network client only:
NOTHING MOVES, no robot SDK. The server (deploy/launch_server_sharpa.sh) must be up on 127.0.0.1:8712.

    ~/venvs/agenticwam-deploy/bin/python scripts/replay_plot.py --episodes 0 100 110 \\
        --token-file runs/deploy_sharpa/token.txt --out runs/replay_plot

Writes <out>/ep<N>.npz (pred, gt, replan_starts) and <out>/replay_compare.png: one column per episode,
top = right-arm joints 1-7, bottom = six right-hand joints; solid = recorded, dashed = predicted.
58-D layout: L arm 0:7 | L hand 7:29 | R arm 29:36 | R hand 36:58.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "deploy/robot"))
import run_sharpa_replay as rp  # noqa: E402
import sharpa_client  # noqa: E402

R_ARM, R_HAND = slice(29, 36), slice(36, 58)


def replay(episode: int, a) -> dict:
    root = a.dataset.resolve()
    instruction = str(pd.read_parquet(root / "meta/tasks.parquet").index[0])
    token = a.token_file.read_text().strip()
    ep = rp.ReplayEpisode(root, episode)
    client = sharpa_client.SharpaPolicyClient(sharpa_client.AgenticWamClientConfig(
        host=a.host, port=a.port, token=token, instruction=instruction,
        request_timeout_s=300.0))  # first /act compiles the denoise core (~75 s)
    pred, starts, t = [], [], 0
    try:
        client.connect()
        client.reset(task=instruction)
        while t < ep.length - 1:
            obs = ep.get_observation(t)
            chunk, info = client.infer(obs, task=instruction)
            rows = min(info["replan_steps"], len(chunk), ep.length - t)
            pred.append(np.asarray(chunk[:rows], np.float64))
            starts.append(t)
            t += rows
    finally:
        client.close()
        ep.close()
    pred = np.concatenate(pred)
    gt = np.asarray(ep.action[:len(pred)], np.float64)
    return {"pred": pred, "gt": gt, "starts": np.array(starts)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 100, 110])
    ap.add_argument("--dataset", type=Path, default=rp.DEFAULT_DATASET)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8712)
    ap.add_argument("--token-file", type=Path, default=REPO / "runs/deploy_sharpa/token.txt")
    ap.add_argument("--out", type=Path, default=REPO / "runs/replay_plot")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {}
    for e in a.episodes:
        r = results[e] = replay(e, a)
        np.savez(a.out / f"ep{e}.npz", **r)
        err = np.abs(r["pred"] - r["gt"])
        print(f"ep {e}: {len(r['pred'])} frames | mean |err| R arm {err[:, R_ARM].mean():.3f} "
              f"R hand {err[:, R_HAND].mean():.3f}", flush=True)

    n = len(a.episodes)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 8), squeeze=False, sharey="row")
    for c, e in enumerate(a.episodes):
        r = results[e]
        frames = np.arange(len(r["pred"]))
        for row, (sl, title, idx) in enumerate(((R_ARM, "right arm joints (rad)", range(7)),
                                               (R_HAND, "right hand joints (first 6)", range(6)))):
            ax = axes[row, c]
            for k, j in enumerate(idx):
                col = f"C{k}"
                ax.plot(frames, r["gt"][:, sl][:, j], color=col, lw=1.4, label=f"j{j + 1} recorded" if c == 0 else None)
                ax.plot(frames, r["pred"][:, sl][:, j], color=col, lw=1.0, ls="--")
            for s in r["starts"]:
                ax.axvline(s, color="0.85", lw=0.5, zorder=0)
            ax.set_title(f"episode {e}: {title}" + ("  [trained on]" if e < 100 else "  [held out]"))
            ax.set_xlabel("frame (30 Hz); grey lines = replan boundaries")
    axes[0, 0].legend(fontsize=7, ncol=2, title="solid = recorded, dashed = predicted")
    fig.tight_layout()
    png = a.out / "replay_compare.png"
    fig.savefig(png, dpi=110)
    print(f"plot: {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
