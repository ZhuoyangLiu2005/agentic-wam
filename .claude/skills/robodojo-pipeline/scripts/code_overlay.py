"""Burn a robot-program panel next to a video.

segments JSON: {"title": str, "subtitle": str, "fps": float,
                "segments": [{"start": frame, "code": str, "label": str?}, ...]}
Frame f shows the last segment whose start <= f. A strip at the bottom marks every segment.
Writes H.264 (yuv420p) through ffmpeg so it plays in browsers.
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
H = int(os.environ.get("OVERLAY_H", 480))   # output video area height
PANEL_W = 620
STRIP_H = 34
BG = (24, 27, 26)
FG = (226, 232, 229)
DIM = (128, 140, 134)
KEY = (110, 200, 185)
SEG_COLORS = [(31, 122, 109), (75, 79, 184), (162, 97, 26), (176, 58, 91), (90, 130, 60), (120, 90, 160)]


def wrap(line, width):
    out = []
    while len(line) > width:
        cut = line.rfind(" ", 0, width)
        cut = cut if cut > width // 2 else width
        out.append(line[:cut])
        line = "    " + line[cut:].lstrip()
    out.append(line)
    return out


def main(video, spec_path, out):
    spec = json.load(open(spec_path))
    segs = sorted(spec["segments"], key=lambda s: s["start"])
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    n = len(frames)
    h0, w0 = frames[0].shape[:2]
    vw = int(round(w0 * H / h0))
    W, HH = vw + PANEL_W, H + STRIP_H
    W += W % 2
    fps = float(spec.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 25)

    f_code = ImageFont.truetype(FONT, 15)
    f_title = ImageFont.truetype(FONT_B, 16)
    f_small = ImageFont.truetype(FONT, 13)
    cw = f_code.getlength("M")
    wrap_w = int((PANEL_W - 36) // cw)

    starts = [s["start"] for s in segs]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{HH}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", os.environ.get("OVERLAY_CRF", "20"), out],
        stdin=subprocess.PIPE)
    for i, fr in enumerate(frames):
        k = max([j for j, s in enumerate(starts) if s <= i] or [0])
        seg = segs[k]
        canvas = Image.new("RGB", (W, HH), BG)
        vid = cv2.cvtColor(cv2.resize(fr, (vw, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        canvas.paste(Image.fromarray(vid), (0, 0))
        d = ImageDraw.Draw(canvas)
        x0, y = vw + 18, 14
        d.text((x0, y), spec.get("title", ""), font=f_title, fill=FG); y += 24
        if spec.get("subtitle"):
            for ln in wrap(spec["subtitle"], int((PANEL_W - 36) // f_small.getlength("M"))):
                d.text((x0, y), ln, font=f_small, fill=DIM); y += 18
        y += 6
        lab = seg.get("label") or f"segment {k + 1}/{len(segs)}"
        counter = seg.get("counter")
        if counter:   # big replan counter; highlighted for `flash_frames` after each replan
            age = i - seg["start"]
            flash = age < int(spec.get("flash_frames", 0)) and seg.get("flash", True)
            fb = ImageFont.truetype(FONT_B, 22)
            if flash:
                d.rectangle([x0 - 8, y - 4, W - 12, y + 30], fill=(176, 58, 91))
            d.text((x0, y), counter + (("   ◀ " + seg.get("flash_text", "REPLAN")) if flash else ""), font=fb, fill=(255, 255, 255) if flash else KEY)
            y += 36
            if flash and age < 6:
                d.rectangle([0, 0, vw - 1, H - 1], outline=(176, 58, 91), width=6)
        d.text((x0, y), f"frame {i + 1}/{n}  ·  {lab}", font=f_small, fill=DIM if counter else KEY); y += 26
        if "subtask" in seg:
            sub = seg["subtask"]
            d.text((x0, y), "subtask", font=f_small, fill=DIM); y += 18
            for ln in wrap(sub, int((PANEL_W - 36) // f_title.getlength("M"))):
                d.text((x0, y), ln, font=f_title, fill=(240, 196, 120) if seg.get("subtask_real", True) else DIM); y += 22
            y += 10
            d.text((x0, y), "code", font=f_small, fill=DIM); y += 20
        d.line([(x0, y - 6), (W - 18, y - 6)], fill=(60, 68, 64))
        for raw in (seg["code"] or "<no code returned>").rstrip("\n").splitlines():
            col = DIM if raw.lstrip().startswith("#") else FG
            for ln in wrap(raw, wrap_w):
                if y > H - 22:
                    d.text((x0, y), "…", font=f_code, fill=DIM)
                    break
                d.text((x0, y), ln, font=f_code, fill=col); y += 20
        # segment strip
        for j, s in enumerate(segs):
            a = int(s["start"] / n * W)
            b = int((segs[j + 1]["start"] if j + 1 < len(segs) else n) / n * W)
            g = s.get("group", j)
            c = SEG_COLORS[g % len(SEG_COLORS)]
            if g != seg.get("group", k):
                c = tuple(int(v * 0.45 + BG[t] * 0.55) for t, v in enumerate(c))
            d.rectangle([a, H + 6, max(a, b - 2), HH - 6], fill=c)
        px = int((i + 0.5) / n * W)
        d.rectangle([px - 1, H + 2, px + 1, HH - 2], fill=(255, 255, 255))
        proc.stdin.write(np.asarray(canvas).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"wrote {out}: {n} frames, {len(segs)} segments, {W}x{HH} @ {fps} fps")


if __name__ == "__main__":
    main(*sys.argv[1:4])
