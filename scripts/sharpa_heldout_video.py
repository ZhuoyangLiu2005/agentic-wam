#!/usr/bin/env python
"""Render the held-out open-loop results of scripts/sharpa_heldout_infer.py as demo videos (CPU only).

One mp4 per episode at the recording's 30 fps. Each frame shows:

    | real head / left wrist / right wrist        |  generated code of the current replan  |
    | imagined head / left wrist / right wrist    |  (header flashes when the code changes) |
    | R EEF xyz | L EEF xyz | R hand | position error per chunk   (GT solid, prediction dashed, cursor) |

The imagined frame shown at offset k into a chunk is video frame min(T-1, round(k / 4)) of that
replan's predicted future (4 action steps per video frame). The prediction plotted is the rows each
replan would execute (the first `replan` rows), joined chunk to chunk.

    python scripts/sharpa_heldout_video.py runs/heldout_openloop/<session> [--episodes 100 110]
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sharpa_heldout_infer import DEFAULT_DATASET, VIDEO_KEYS, load_episode  # noqa: E402

FPS = 30
VIDEO_FREQ_RATIO = 4          # action steps per imagined video frame (data.train.action_video_freq_ratio)
VIEW_H, VIEW_W = 224, 320
LABEL_H = 22
CODE_W = 640
PLOT_H = 360
FLASH_FRAMES = 15             # header highlight after a code change
CAM_NAMES = {"head_camera": "head", "left_camera": "left wrist", "right_camera": "right wrist"}


def _font(size: int):
    from PIL import ImageFont
    import matplotlib
    path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono.ttf"
    return ImageFont.truetype(str(path), size)


def _label(img, text: str, font, bg=(30, 30, 30), fg=(235, 235, 235)):
    """Draw a label bar across the top of a PIL image region."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, LABEL_H], fill=bg)
    d.text((6, 3), text, font=font, fill=fg)


def stitched_prediction(t: np.ndarray, pred: np.ndarray, replan: int, n: int) -> np.ndarray:
    """[R,H,64] chunks -> [n,64] trajectory of the executed rows (NaN where nothing was predicted)."""
    traj = np.full((n, pred.shape[-1]), np.nan)
    for ti, chunk in zip(t, pred):
        k = min(replan, len(chunk), n - ti)
        traj[ti:ti + k] = chunk[:k]
    return traj


def plot_background(npz, replan: int, width: int):
    """Static episode plots (matplotlib, drawn once) + a function mapping frame -> cursor x pixels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = int(npz["length"])
    gt = npz["gt_abs64_full"]
    pred = stitched_prediction(npz["t"], npz["pred_abs64"], replan, n)
    x = np.arange(n) / FPS
    dpi = 100
    fig, axes = plt.subplots(1, 4, figsize=(width / dpi, PLOT_H / dpi), dpi=dpi)
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    for ax, (title, sl) in zip(axes[:2], (("right EEF xyz (m)", slice(32, 35)), ("left EEF xyz (m)", slice(0, 3)))):
        for i, c in enumerate(colors):
            ax.plot(x, gt[:, sl][:, i], color=c, lw=1.2, label="xyz"[i] + " GT")
            ax.plot(x, pred[:, sl][:, i], color=c, lw=1.2, ls="--", label="xyz"[i] + " pred")
        ax.set_title(title, fontsize=9)
    ax = axes[2]
    ax.plot(x, gt[:, 41:63].mean(-1), color="k", lw=1.2, label="GT")
    ax.plot(x, pred[:, 41:63].mean(-1), color="#ff7f0e", lw=1.2, ls="--", label="pred")
    ax.set_title("right hand, mean of 22 joints (rad)", fontsize=9)
    ax = axes[3]
    t = npz["t"]
    for side, eef, c in (("L", slice(0, 3), "#9467bd"), ("R", slice(32, 35), "#d62728")):
        err = np.linalg.norm(npz["pred_abs64"][:, :replan, eef] - npz["gt_abs64"][:, :replan, eef], axis=-1) * 1e3
        ax.step(t / FPS, np.nanmean(err, axis=1), where="post", color=c, lw=1.2, label=f"{side} arm")
    ax.set_title("EEF position error per chunk (mm)", fontsize=9)
    for ax in axes:
        ax.set_xlim(0, n / FPS)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("s", fontsize=7)
        ax.legend(fontsize=6, ncol=2, loc="best")
        for ti in t:
            ax.axvline(ti / FPS, color="0.85", lw=0.5, zorder=0)
    fig.tight_layout(pad=0.6)
    fig.canvas.draw()
    bg = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    h = bg.shape[0]
    spans = []   # per axis: (x pixel at time 0, x pixel at end, y top, y bottom) in image coordinates
    for ax in axes:
        (x0, y0), (x1, y1) = ax.transData.transform([(0, ax.get_ylim()[0]), (n / FPS, ax.get_ylim()[1])])
        spans.append((x0, x1, h - y1, h - y0))
    plt.close(fig)
    return bg, spans


def code_panel(code: str, header: str, flash: bool, font, small_font):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (CODE_W, 2 * (VIEW_H + LABEL_H)), (250, 250, 247))
    d = ImageDraw.Draw(img)
    _label(img, header, small_font, bg=(200, 120, 0) if flash else (30, 30, 30))
    y, line_h = LABEL_H + 6, 16
    chars = int((CODE_W - 12) / font.getlength("M")) - 1
    for raw in (code or "<no code>").splitlines():
        for line in textwrap.wrap(raw, chars, subsequent_indent="      ", drop_whitespace=False) or [""]:
            if y > img.height - line_h:
                d.text((6, y), "...", font=font, fill=(150, 0, 0))
                return img
            d.text((6, y), line, font=font, fill=(20, 20, 20))
            y += line_h
    return img


def render_episode(ep_dir: Path, shard: Path, replan: int) -> Path:
    import cv2
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from agenticwam.utils.video_io import save_mp4

    episode = int(ep_dir.name.split("_")[1])
    npz = np.load(ep_dir / "pred.npz")
    recs = [json.loads(l) for l in (ep_dir / "codes.jsonl").read_text().splitlines() if l.strip()]
    ep = load_episode(shard, episode)
    n = ep["length"]
    t = npz["t"]
    gen = npz["gen_video"]                         # [R, nc, T, H, W, 3]
    last = min(n, int(t[-1]) + replan)
    width = 3 * VIEW_W + CODE_W
    bg, spans = plot_background(npz, replan, width)
    font, small = _font(13), _font(13)
    height = -(-(2 * (VIEW_H + LABEL_H) + bg.shape[0]) // 16) * 16   # x264 macro blocks: no resize

    def frames():
        r, panel = -1, None
        for f in range(last):
            while r + 1 < len(t) and t[r + 1] <= f:
                r += 1
            k = f - int(t[r])
            if k == 0 or panel is None:
                changed = r > 0 and recs[r]["code"] != recs[r - 1]["code"]
                tag = "new code" if changed or r == 0 else "same code"
                header = f"ep {episode} | replan {r} | t={t[r]} ({t[r] / FPS:.1f}s) | {tag}"
                panel_on = code_panel(recs[r]["code"], header, True, font, small) if changed else None
                panel = code_panel(recs[r]["code"], header, False, font, small)
            canvas = Image.new("RGB", (width, height), (255, 255, 255))
            vi = min(gen.shape[2] - 1, int(round(k / VIDEO_FREQ_RATIO)))
            for j, cam in enumerate(VIDEO_KEYS):
                real = Image.new("RGB", (VIEW_W, VIEW_H + LABEL_H))
                real.paste(Image.fromarray(ep[cam][f]), (0, LABEL_H))
                _label(real, f"real {CAM_NAMES[cam]}  {f / FPS:5.1f}s", small)
                imag = Image.new("RGB", (VIEW_W, VIEW_H + LABEL_H))
                imag.paste(Image.fromarray(gen[r, j, vi]), (0, LABEL_H))
                _label(imag, f"imagined {CAM_NAMES[cam]}  replan {r} +{vi}", small, bg=(40, 40, 110))
                canvas.paste(real, (j * VIEW_W, 0))
                canvas.paste(imag, (j * VIEW_W, VIEW_H + LABEL_H))
            canvas.paste(panel_on if (panel_on is not None and k < FLASH_FRAMES) else panel, (3 * VIEW_W, 0))
            plot = bg.copy()
            for x0, x1, y_top, y_bot in spans:
                cx = int(round(x0 + (x1 - x0) * f / n))
                cv2.line(plot, (cx, int(y_top)), (cx, int(y_bot)), (0, 0, 0), 1)
            canvas.paste(Image.fromarray(plot), (0, 2 * (VIEW_H + LABEL_H)))
            yield canvas

    out = ep_dir / "demo.mp4"
    save_mp4(frames(), str(out), fps=FPS)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path, help="runs/heldout_openloop/<session>")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="default: every ep_* in the session")
    a = ap.parse_args()
    meta = json.loads((a.session / "session.json").read_text())
    shard = Path(meta.get("dataset", DEFAULT_DATASET)) / "train_shard_0"
    dirs = sorted(a.session.glob("ep_*"))
    if a.episodes:
        dirs = [d for d in dirs if int(d.name.split("_")[1]) in a.episodes]
    for d in dirs:
        if not (d / "pred.npz").exists():
            print(f"skip {d.name}: no pred.npz")
            continue
        print(f"wrote {render_episode(d, shard, int(meta['replan']))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
