#!/usr/bin/env python3
"""Summarise server-replay results (scripts/replay_plot.py ep<N>.npz) per checkpoint and episode.

    python scripts/replay_summary.py [--out runs/replay_summary.md]

Mean |predicted - recorded| in rad: right arm = 58-D idx 29:36, right hand = idx 36:58. Reads the step-5000/10000/15000
dirs under runs/replay_plot{,_more}/run_20k_step*/ and step 20000 (runs/replay_plot/ep*.npz + replay_plot_more/tong-transfer-sharpa_step20000).
Plain numpy: no GPU, no network.
"""
import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
SOURCES = {  # step -> dirs holding ep<N>.npz
    5000: [RUNS / "replay_plot/run_20k_step5000", RUNS / "replay_plot_more/run_20k_step5000"],
    10000: [RUNS / "replay_plot/run_20k_step10000", RUNS / "replay_plot_more/run_20k_step10000"],
    15000: [RUNS / "replay_plot_more/run_20k_step15000"],
    20000: [RUNS / "replay_plot", RUNS / "replay_plot_more/tong-transfer-sharpa_step20000"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RUNS / "replay_summary.md")
    a = ap.parse_args()
    res = {}  # (step, ep) -> (arm, hand)
    for step, dirs in SOURCES.items():
        for d in dirs:
            for f in sorted(d.glob("ep*.npz")):
                r = np.load(f)
                err = np.abs(r["pred"] - r["gt"])
                res[(step, int(f.stem[2:]))] = (err[:, 29:36].mean(), err[:, 36:58].mean())
    eps = sorted({e for _, e in res})
    steps = sorted({s for s, _ in res})
    lines = []
    for name, k in (("right arm", 0), ("right hand", 1)):
        lines += [f"**{name}** mean |err| (rad)", "",
                  "| step | " + " | ".join(f"ep {e}{'*' if e < 100 else ''}" for e in eps) + " | held-out mean |",
                  "|---|" + "---|" * (len(eps) + 1)]
        for s in steps:
            row = [res.get((s, e)) for e in eps]
            held = [r[k] for e, r in zip(eps, row) if r and e >= 100]
            lines.append(f"| {s} | " + " | ".join(f"{r[k]:.3f}" if r else "-" for r in row)
                         + f" | {np.mean(held):.3f} |" if held else " | - |")
        lines.append("")
    lines.append("`*` = trained on (episodes 0-99); held-out mean is over episodes >= 100.")
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
