"""Robot state at policy calls: RoboDojo sim eval (AGENTICWAM_STATE_LOG) vs the task's training shard.

args: state_log.jsonl out_dir
State is 14-D [L q0..5, L grip, R q0..5, R grip]; training rows come from the 82-D shard ([34:40] L q, [32] L grip,
[40:46] R q, [33] R grip), sampled every 24 frames like the policy calls.
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rd_env import REPO, SHARD, SRC, TASK, TRAIN_EPS  # noqa: E402
NAMES = [f"L q{i}" for i in range(6)] + ["L grip"] + [f"R q{i}" for i in range(6)] + ["R grip"]
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3df"


def to14(s82):
    return np.concatenate([s82[:, 34:40], s82[:, 32:33], s82[:, 40:46], s82[:, 33:34]], 1)


def main(log, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    trained = set(json.load(open(TRAIN_EPS)))
    tr_eps = {}
    for p in sorted(glob.glob(str(SHARD / "data/chunk-000/episode_*.parquet"))):
        ep = int(Path(p).stem.split("_")[1])
        if ep in trained:
            tr_eps[ep] = to14(np.stack(pd.read_parquet(p)["observation.state"]))
    tr_all = np.concatenate(list(tr_eps.values()))
    tr_calls = np.concatenate([v[::24] for v in tr_eps.values()])

    ev = defaultdict(list)
    for line in open(log):
        r = json.loads(line); ev[r["episode"]].append(r)
    ev_eps = {}
    track = []
    for e, rows in sorted(ev.items()):
        rows.sort(key=lambda r: r["chunk"])
        st = np.array([r["state"] for r in rows])
        if len(st) < 3:          # the server's spurious second reset gives an empty/1-call episode
            continue
        ev_eps[e] = st
        for a, b in zip(rows[:-1], rows[1:]):   # last executed target of call k vs state observed at call k+1
            track.append(np.array(b["state"]) - np.array(a["targets"])[-1])
    ev_all = np.concatenate(list(ev_eps.values()))
    track = np.array(track)

    # 1. per-dimension ranges
    lo, hi = np.percentile(tr_all, 1, 0), np.percentile(tr_all, 99, 0)
    sd = tr_all.std(0) + 1e-6
    rep = {"n_train_frames": int(len(tr_all)), "n_eval_calls": int(len(ev_all)), "episodes": sorted(ev_eps), "dims": {}}
    for i, n in enumerate(NAMES):
        x = ev_all[:, i]
        rep["dims"][n] = {
            "train_p1_p99": [round(float(lo[i]), 3), round(float(hi[i]), 3)],
            "eval_min_max": [round(float(x.min()), 3), round(float(x.max()), 3)],
            "eval_frac_outside_p1_p99": round(float(((x < lo[i]) | (x > hi[i])).mean()), 3),
            "tracking_err_mean_abs": round(float(np.abs(track[:, i]).mean()), 4),
            "tracking_err_p95_abs": round(float(np.percentile(np.abs(track[:, i]), 95)), 4),
        }
    # 2. grippers: value histogram
    for i, n in ((6, "L grip"), (13, "R grip")):
        rep["dims"][n]["train_unique_rounded"] = sorted(set(np.round(tr_all[:, i], 2).tolist()))[:12]
        rep["dims"][n]["eval_unique_rounded"] = sorted(set(np.round(ev_all[:, i], 2).tolist()))[:12]
    # 3. nearest-neighbour distance (z-scored 14-D) of each eval call to any training frame, vs train->train (leave-episode-out)
    def nn(q, ref):
        qz, rz = q / sd, ref / sd
        d = ((qz[:, None, :] - rz[None, :, :]) ** 2).sum(-1)
        return np.sqrt(d.min(1))
    nn_ev = {e: nn(v, tr_all) for e, v in ev_eps.items()}
    nn_tr = []
    eps = list(tr_eps)
    rng = np.random.default_rng(0)
    for e in rng.choice(eps, size=min(10, len(eps)), replace=False):
        ref = np.concatenate([tr_eps[k] for k in eps if k != e])
        nn_tr.append(nn(tr_eps[e][::24], ref))
    nn_tr = np.concatenate(nn_tr)
    rep["nn_dist_z"] = {"train_heldout_episode_median": round(float(np.median(nn_tr)), 3),
                        "train_heldout_episode_p95": round(float(np.percentile(nn_tr, 95)), 3),
                        "eval_median": round(float(np.median(np.concatenate(list(nn_ev.values())))), 3),
                        "eval_p95": round(float(np.percentile(np.concatenate(list(nn_ev.values())), 95)), 3)}
    (out / "state_compare.json").write_text(json.dumps(rep, indent=1) + "\n")

    # figure: per-dim distributions + NN distance over time
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 7, height_ratios=[1, 1, 0.15, 1.3])
    for i, n in enumerate(NAMES):
        ax = fig.add_subplot(gs[i // 7, i % 7])
        b = np.linspace(min(tr_all[:, i].min(), ev_all[:, i].min()), max(tr_all[:, i].max(), ev_all[:, i].max()), 40)
        ax.hist(tr_calls[:, i], bins=b, density=True, color=BLUE, alpha=0.55, label="training")
        ax.hist(ev_all[:, i], bins=b, density=True, color=ORANGE, alpha=0.55, label="eval")
        ax.set_title(n, color=INK, fontsize=9); ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        if i == 0:
            ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(gs[3, :])
    for e, d in nn_ev.items():
        ax.plot(np.arange(len(d)), d, color=ORANGE, lw=1.6, alpha=0.8)
    ax.axhline(np.percentile(nn_tr, 95), color=BLUE, lw=1.5, ls="--",
               label=f"training, held-out episode p95 = {np.percentile(nn_tr, 95):.2f}")
    ax.axhline(np.median(nn_tr), color=BLUE, lw=1.5, label=f"training, held-out episode median = {np.median(nn_tr):.2f}")
    ax.plot([], [], color=ORANGE, lw=1.6, label="eval episodes")
    ax.set_xlabel("policy call (every 24 steps, ~1 s)"); ax.set_ylabel("distance to nearest training state (z)")
    ax.set_ylim(bottom=0); ax.grid(axis="y", color=GRID); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.suptitle("Robot state at policy calls · training (training shard) vs sim eval · step20000",
                 x=0.01, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "state_compare.png", dpi=140, facecolor="white")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
