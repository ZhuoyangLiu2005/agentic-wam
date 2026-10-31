#!/usr/bin/env python3
"""EEF error of the server replay: FK of the predicted vs the recorded joint rows (scripts/replay_plot.py ep<N>.npz).

    ~/venvs/agenticwam-deploy/bin/python scripts/replay_eef_error.py [--out docs/images/eef_by_episode] [--md runs/replay_eef_summary.md]

Both 58-D streams ([L arm 7, L hand 22, R arm 7, R hand 22]) go through the deployment FK (utils/vega_fk_ik/vega_kinematics.py,
torso/head as in conversion) -> L_ee / R_ee 4x4 poses in the world frame. Per frame: position error = |p_pred - p_gt| (mm),
rotation error = geodesic angle of R_pred^T R_gt (deg). Needs pinocchio (deploy venv); no GPU, no robot.
Writes one figure per episode (columns = checkpoints, rows = right-EEF position / rotation, thin = per frame,
thick = 1 s moving average, grey = replan boundaries) and a markdown table of per-episode means.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "utils/vega_fk_ik"))
from replay_summary import SOURCES  # noqa: E402
import vega_kinematics as vk  # noqa: E402

L_ARM, R_ARM = slice(0, 7), slice(29, 36)


def poses(rows):
    """[N,58] joint rows -> (T_L [N,4,4], T_R [N,4,4])."""
    TL, TR = [], []
    for r in rows:
        tl, tr = vk.fk_both(r[L_ARM], r[R_ARM])
        TL.append(tl)
        TR.append(tr)
    return np.stack(TL), np.stack(TR)


def errors(Tp, Tg):
    """Per-frame position error (mm) and geodesic rotation error (deg) between two pose streams."""
    pos = np.linalg.norm(Tp[:, :3, 3] - Tg[:, :3, 3], axis=1) * 1000.0
    Rrel = np.einsum("nji,njk->nik", Tp[:, :3, :3], Tg[:, :3, :3])
    cos = np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return pos, np.degrees(np.arccos(cos))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "docs/images/eef_by_episode")
    ap.add_argument("--md", type=Path, default=REPO / "runs/replay_eef_summary.md")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = {}
    for step, dirs in SOURCES.items():
        for d in dirs:
            for f in d.glob("ep*.npz"):
                files[(step, int(f.stem[2:]))] = f
    steps = sorted({s for s, _ in files})
    eps = sorted({e for _, e in files})
    res, curves = {}, {}
    for (s, e), f in sorted(files.items()):
        r = np.load(f)
        TLp, TRp = poses(r["pred"])
        TLg, TRg = poses(r["gt"])
        pR, rR = errors(TRp, TRg)
        pL, rL = errors(TLp, TLg)
        curves[(s, e)] = (pR, rR, r["starts"])
        res[(s, e)] = (pR.mean(), rR.mean(), pL.mean(), rL.mean())
        print(f"step {s} ep {e}: R {pR.mean():.1f} mm {rR.mean():.2f} deg | L {pL.mean():.1f} mm {rL.mean():.2f} deg", flush=True)

    win = np.ones(30) / 30
    for e in eps:
        have = [s for s in steps if (s, e) in curves]
        fig, axes = plt.subplots(2, len(have), figsize=(5.2 * len(have), 6), squeeze=False, sharey="row")
        for c, s in enumerate(have):
            pR, rR, starts = curves[(s, e)]
            for row, (y, unit, name) in enumerate(((pR, "mm", "right EEF position error"), (rR, "deg", "right EEF rotation error"))):
                ax = axes[row, c]
                ax.plot(y, color="C0", lw=0.5, alpha=0.5)
                ax.plot(np.convolve(y, win, mode="same"), color="C3", lw=1.6)
                for t in starts:
                    ax.axvline(t, color="0.88", lw=0.4, zorder=0)
                ax.set_title(f"step {s}: {name}\nmean {y.mean():.1f} {unit}" if row == 0 else
                             f"step {s}: {name}\nmean {y.mean():.2f} {unit}", fontsize=10)
                ax.set_xlabel("frame (30 Hz)")
                ax.set_ylabel(unit)
        fig.suptitle(f"episode {e} [{'trained on' if e < 100 else 'held out'}]  thin = per frame, red = 1 s moving average", fontsize=12)
        fig.tight_layout()
        fig.savefig(a.out / f"ep{e}.png", dpi=90)
        plt.close(fig)

    lines = []
    for name, idx, fmt in (("right EEF position error (mm)", 0, "{:.1f}"), ("right EEF rotation error (deg)", 1, "{:.2f}"),
                           ("left EEF position error (mm)", 2, "{:.1f}"), ("left EEF rotation error (deg)", 3, "{:.2f}")):
        lines += [f"**{name}**", "", "| step | " + " | ".join(f"ep {e}{'*' if e < 100 else ''}" for e in eps) + " | held-out mean |",
                  "|---|" + "---|" * (len(eps) + 1)]
        for s in steps:
            row = [res.get((s, e)) for e in eps]
            held = [r[idx] for e, r in zip(eps, row) if r and e >= 100]
            lines.append(f"| {s} | " + " | ".join(fmt.format(r[idx]) if r else "-" for r in row) + f" | {np.mean(held):{fmt[2:-1]}} |")
        lines.append("")
    lines.append("`*` = trained on (episodes 0-99); held-out mean is over episodes >= 100.")
    a.md.parent.mkdir(parents=True, exist_ok=True)
    a.md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
