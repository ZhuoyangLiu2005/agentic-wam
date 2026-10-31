#!/usr/bin/env python3
"""One figure per episode with the checkpoints side by side (reads the saved scripts/replay_plot.py ep<N>.npz).

    python scripts/replay_compare_by_episode.py [--out docs/images/by_episode]

Columns = checkpoints (5000, 10000, 15000, 20000), rows = right-arm joints 1-7 and the first 6 right-hand joints;
solid = recorded, dashed = predicted, y axes shared across the figure. Writes <out>/ep<N>.png. Numpy + matplotlib only.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_summary import SOURCES  # noqa: E402

R_ARM, R_HAND = slice(29, 36), slice(36, 58)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/images/by_episode"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = {}  # (step, ep) -> npz
    for step, dirs in SOURCES.items():
        for d in dirs:
            for f in d.glob("ep*.npz"):
                files[(step, int(f.stem[2:]))] = f
    steps = sorted({s for s, _ in files})
    for e in sorted({e for _, e in files}):
        have = [s for s in steps if (s, e) in files]
        fig, axes = plt.subplots(2, len(have), figsize=(5.2 * len(have), 7), squeeze=False, sharey="row")
        for c, s in enumerate(have):
            r = np.load(files[(s, e)])
            err = np.abs(r["pred"] - r["gt"])
            x = np.arange(len(r["pred"]))
            for row, (sl, name, n) in enumerate(((R_ARM, "right arm joints (rad)", 7), (R_HAND, "right hand joints 1-6 (rad)", 6))):
                ax = axes[row, c]
                for j in range(n):
                    ax.plot(x, r["gt"][:, sl][:, j], color=f"C{j}", lw=1.3, label=f"j{j + 1}" if c == 0 else None)
                    ax.plot(x, r["pred"][:, sl][:, j], color=f"C{j}", lw=0.9, ls="--")
                for t in r["starts"]:
                    ax.axvline(t, color="0.88", lw=0.4, zorder=0)
                m = err[:, sl].mean()
                ax.set_title(f"step {s}: {name}\nmean |err| {m:.3f}", fontsize=10)
                ax.set_xlabel("frame (30 Hz)")
        axes[0, 0].legend(fontsize=7, ncol=4, title="solid = recorded, dashed = predicted")
        fig.suptitle(f"episode {e} [{'trained on' if e < 100 else 'held out'}]", fontsize=13)
        fig.tight_layout()
        fig.savefig(a.out / f"ep{e}.png", dpi=90)
        plt.close(fig)
        print(a.out / f"ep{e}.png", have)


if __name__ == "__main__":
    main()
