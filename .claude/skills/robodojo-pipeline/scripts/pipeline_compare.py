"""Image pipeline parity: training data path vs inference path, on the SAME source frame.

Source = the RoboDojo LeRobot v2.1 AV1 480x640 frame (what the sim camera rendered at data-collection time).
  T (training):  source -> converter ffmpeg (crop 16:464, scale area, h264 yuv420p) -> shard mp4
                 -> torchcodec decode (LeRobot loader) -> ToTensor -> torchvision Resize(224,320)
                 -> ResizeSmallestSideAspectPreserving -> CenterCrop -> Normalize(0.5,0.5)
  I (inference): source RGB -> model.training_view (crop 16:464, cv2 INTER_AREA) -> deploy_policy._build_obs_video
                 (/255, tv resize [224,320] antialias, ResizeSmallestSide, CenterCrop, Normalize)
Each stage's tensor shape and the T-vs-I difference are printed.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rd_env import REPO, SHARD, SRC, TASK, TRAIN_EPS  # noqa: E402
sys.path.insert(0, str(REPO / "src"))
CAMS = [("observation.images.cam_high", "cam_high"), ("observation.images.cam_left_wrist", "cam_left_wrist"),
        ("observation.images.cam_right_wrist", "cam_right_wrist")]


def torchcodec_frames(path, idx):
    from torchcodec.decoders import VideoDecoder
    d = VideoDecoder(str(path))
    return d.get_frames_at(indices=list(idx)).data      # [N,3,H,W] uint8 RGB


def stage2(video):
    from agenticwam.datasets.lerobot.transforms.image import ToTensor  # noqa: F401  (import check)
    from agenticwam.datasets.utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize  # type: ignore
    return video


def main():
    eps = []
    for line in (SRC / "meta/episodes.jsonl").read_text().splitlines():
        r = json.loads(line)
        eps.append(r["episode_index"])
    import pandas as pd
    info = json.loads((SRC / "meta/info.json").read_text())
    chunk_size = info["chunks_size"]
    # this task's source episodes: scan parquet task_index (the converter does the same, sorted)
    tasks_of = {}
    for e in eps:
        p = SRC / f"data/chunk-{e // chunk_size:03d}/episode_{e:06d}.parquet"
        tasks_of[e] = int(pd.read_parquet(p, columns=["task_index"])["task_index"].iloc[0])
    task_eps = sorted(e for e, t in tasks_of.items() if t == TASK)
    shard_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    src_ep = task_eps[shard_ep]
    frames = [0, 60, 120]
    print(f"shard episode {shard_ep} <- source episode {src_ep} (task {TASK}); frames {frames}")

    sys.path.insert(0, str(REPO / "deploy/robodojo/AgenticWAM"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("aw_model_tv", REPO / "deploy/robodojo/AgenticWAM/model.py")
    src_txt = (REPO / "deploy/robodojo/AgenticWAM/model.py").read_text()
    ns = {"np": np, "cv2": cv2}
    exec(src_txt[src_txt.index("SRC_H, SRC_W"):src_txt.index("def _load_policy_runtime")].replace(
        "def _none_like", "def _none_like_"), ns)
    training_view = ns["training_view"]

    from torchvision.transforms.functional import resize as tv_resize
    import torchvision.transforms as T
    from agenticwam.datasets.lerobot.transforms.image import ToTensor
    mod = sys.modules.get("agenticwam.datasets.lerobot.robot_video_dataset")
    import agenticwam.datasets.lerobot.robot_video_dataset as rvd
    RS, CC, NM = rvd.ResizeSmallestSideAspectPreserving, rvd.CenterCrop, rvd.Normalize
    rs, cc, nm = RS(args={"img_w": 320, "img_h": 224}), CC(args={"img_w": 320, "img_h": 224}), NM(args={"mean": 0.5, "std": 0.5})
    train_tf = [ToTensor(), T.Resize([224, 320])]

    def stats(name, a, b):
        d = (a.float() - b.float())
        print(f"  {name:46s} T{tuple(a.shape)} I{tuple(b.shape)}  mean|d|={d.abs().mean():.4f}  max|d|={d.abs().max():.4f}  "
              f"mean d per ch={[round(float(x), 4) for x in d.mean(dim=tuple(i for i in range(d.ndim) if i != d.ndim - 3))]}")

    for key, cam in CAMS:
        src_mp4 = SRC / f"videos/chunk-{src_ep // chunk_size:03d}/{key}/episode_{src_ep:06d}.mp4"
        shard_mp4 = SHARD / f"videos/chunk-000/{key}/episode_{shard_ep:06d}.mp4"
        src = torchcodec_frames(src_mp4, frames)                       # [N,3,480,640]
        tr = torchcodec_frames(shard_mp4, frames)                      # [N,3,224,320]
        print(f"\n[{cam}] source {tuple(src.shape)} {src_mp4.suffix}  shard {tuple(tr.shape)}")
        # inference stage 1: training_view on HWC uint8
        inf = torch.stack([torch.from_numpy(training_view(f.permute(1, 2, 0).numpy())).permute(2, 0, 1) for f in src])
        stats("uint8 224x320 (shard decode vs training_view)", tr, inf)
        # training: ToTensor + Resize
        t = tr
        for f in train_tf:
            t = f(t)
        i = inf.float() / 255.0
        i_s1 = tv_resize(i, [224, 320], antialias=True)
        stats("after train_transforms / stage1 resize", t, i_s1)
        t2, i2 = nm(cc(rs(t))), nm(cc(rs(i_s1)))
        stats("final model input [-1,1]", t2, i2)
        # decomposition: converter's ffmpeg vs cv2 area on identical pixels, without h264 (lossless rgb)
        if cam == "cam_high":
            import subprocess, tempfile
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "x.mkv"
                subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src_mp4), "-vf",
                                "crop=640:448:0:16,scale=320:224:flags=area", "-frames:v", "121",
                                "-c:v", "ffv1", "-pix_fmt", "yuv444p", str(out)], check=True)
                ff = torchcodec_frames(out, frames)
                out2 = Path(td) / "y.mkv"
                subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src_mp4), "-vf",
                                "crop=640:448:0:16,scale=320:224:flags=area,format=rgb24", "-frames:v", "121",
                                "-c:v", "ffv1", "-pix_fmt", "bgr0", str(out2)], check=True)
                ff_rgb = torchcodec_frames(out2, frames)
            stats("decomp: ffmpeg area (yuv444, lossless) vs cv2", ff, inf)
            stats("decomp: ffmpeg area (rgb, lossless) vs cv2", ff_rgb, inf)
            stats("decomp: h264 yuv420 shard vs lossless ffmpeg", tr, ff)
            for k, f in enumerate(frames):
                cv2.imwrite(str(Path(sys.argv[2]) / f"pipe_{cam}_f{f}.png"), np.concatenate(
                    [tr[k].permute(1, 2, 0).numpy(), inf[k].permute(1, 2, 0).numpy(),
                     np.clip(np.abs(tr[k].int() - inf[k].int()) * 8, 0, 255).permute(1, 2, 0).numpy().astype(np.uint8)],
                    1)[..., ::-1])


if __name__ == "__main__":
    main()
