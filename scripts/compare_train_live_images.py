#!/usr/bin/env python
"""Compare the camera frames the model was TRAINED on with the frames it gets LIVE -- read-only, NOTHING MOVES.

Training frames: the 224x320 per-episode videos of the v2.1 dataset (what the model saw in training).
Live frames: the "model view" half of a preview jpg written by deploy/robot/run_sharpa_eval.py
(`--preview-only` -> <session>/preview_model_view.jpg, or a real run -> ep_NNN/obs/replan_NNN.jpg).
Those jpgs are 3 cameras wide; top row = raw frames downsized to 320x180, bottom row = the exact 320x224 model view.

    python scripts/compare_train_live_images.py \\
        --dataset data_local/trex_tong_transfer_aw76_v21 \\
        --live runs/sharpa_eval/<session>/preview_model_view.jpg [--episode 0 --frame 300]

Writes <out>/compare_images.png (rows: training, live; columns: head, left wrist, right wrist) and prints,
per camera, the per-channel mean/std of the live frame against the distribution of the same statistic over
training frames (z-score), plus an RGB-vs-BGR check. Large |z| (> 3) on a camera means the live image does not
look like the training images (exposure, white balance, framing, channel order).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sharpa_heldout_infer import VIDEO_KEYS, load_episode  # noqa: E402

CAMS = list(VIDEO_KEYS)            # head_camera, left_camera, right_camera
H, W = 224, 320                    # model view
RAW_ROW_H = 180                    # preview_grid: raw frames resized to 320x180 on top


def split_live_model_view(jpg: Path) -> dict:
    """preview jpg (BGR on disk) -> {cam: [224,320,3] uint8 RGB} from the bottom (model-view) row."""
    import cv2
    img = cv2.imread(str(jpg))
    if img is None:
        raise SystemExit(f"cannot read {jpg}")
    if img.shape[:2] != (RAW_ROW_H + H, 3 * W):
        raise SystemExit(f"{jpg} is {img.shape[1]}x{img.shape[0]}; expected {3 * W}x{RAW_ROW_H + H} "
                         "(a preview_grid jpg from run_sharpa_eval.py)")
    rgb = cv2.cvtColor(img[RAW_ROW_H:], cv2.COLOR_BGR2RGB)
    return {cam: rgb[:, i * W:(i + 1) * W] for i, cam in enumerate(CAMS)}


def frame_stats(frame: np.ndarray) -> np.ndarray:
    """[H,W,3] -> [6] = RGB channel means then stds (0-255)."""
    f = frame.reshape(-1, 3).astype(np.float64)
    return np.concatenate([f.mean(0), f.std(0)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=REPO / "data_local/trex_tong_transfer_aw76_v21")
    ap.add_argument("--live", type=Path, required=True, help="preview jpg from run_sharpa_eval.py")
    ap.add_argument("--episode", type=int, default=0, help="training episode shown in the picture")
    ap.add_argument("--frame", type=int, default=0, help="frame of that episode shown in the picture")
    ap.add_argument("--stat-episodes", type=int, nargs="+", default=[0, 25, 50, 75],
                    help="training episodes whose frames give the reference distribution")
    ap.add_argument("--stat-frames", type=int, default=40, help="evenly spaced frames per stat episode")
    ap.add_argument("--out", type=Path, default=REPO / "runs/compare_obs")
    a = ap.parse_args()

    shard = a.dataset.resolve() / "train_shard_0"
    live = split_live_model_view(a.live)

    shown = load_episode(shard, a.episode)
    if not 0 <= a.frame < shown["length"]:
        raise SystemExit(f"episode {a.episode} has {shown['length']} frames")

    # reference distribution of per-frame channel stats over training frames
    ref = {cam: [] for cam in CAMS}
    for ep_i in a.stat_episodes:
        ep = shown if ep_i == a.episode else load_episode(shard, ep_i)
        idx = np.linspace(0, ep["length"] - 1, a.stat_frames).astype(int)
        for cam in CAMS:
            ref[cam].extend(frame_stats(ep[cam][i]) for i in idx)
    ref = {cam: np.stack(v) for cam, v in ref.items()}

    names = ["R mean", "G mean", "B mean", "R std", "G std", "B std"]
    print(f"live: {a.live}\ntraining reference: episodes {a.stat_episodes} x {a.stat_frames} frames\n")
    suspicious = False
    for cam in CAMS:
        mu, sd = ref[cam].mean(0), ref[cam].std(0).clip(min=1e-6)
        s = frame_stats(live[cam])
        z = (s - mu) / sd
        print(f"{cam}")
        for n, sv, m, d, zv in zip(names, s, mu, sd, z):
            flag = "  <-- OUT OF RANGE" if abs(zv) > 3 else ""
            print(f"  {n}: live {sv:6.1f} | train {m:6.1f} +/- {d:5.1f} | z {zv:+5.1f}{flag}")
        suspicious |= bool((np.abs(z) > 3).any())
        # channel order: is the live frame closer to training with R and B swapped?
        d_rgb = np.abs(s[:3] - mu[:3]).sum()
        d_bgr = np.abs(s[:3][::-1] - mu[:3]).sum()
        if d_bgr < 0.7 * d_rgb:
            print("  !! live channel means match training better with R and B swapped (RGB/BGR mix-up?)")
            suspicious = True
    print("\nResult:", "live frames DIFFER from training in the flagged statistics" if suspicious
          else "live frame statistics are within the training range (necessary, not sufficient: look at the picture)")

    import cv2
    rows = [np.hstack([shown[cam][a.frame] for cam in CAMS]), np.hstack([live[cam] for cam in CAMS])]
    grid = np.vstack(rows)
    for r, label in enumerate((f"TRAIN ep{a.episode} frame{a.frame}", "LIVE")):
        cv2.putText(grid, label, (6, 18 + r * H), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "compare_images.png"
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"picture: {path}  (top: training, bottom: live; head | left wrist | right wrist)")
    (a.out / "compare_images.json").write_text(json.dumps(
        {"live": str(a.live), "stat_episodes": a.stat_episodes, "suspicious": suspicious}, indent=2))
    return 1 if suspicious else 0


if __name__ == "__main__":
    sys.exit(main())
