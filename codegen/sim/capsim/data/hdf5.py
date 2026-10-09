"""RoboTwin saver hdf5 access. The saver cv2.imencode()s RGB arrays, so the stored JPEGs are BGR:
`decode_robotwin_jpeg` is the one place that swaps channels on decode."""
import io
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from capsim.profiles import DEFAULT_EMBODIMENT, load_profile

# no env here: the embodiment comes from CAPSIM_EMBODIMENT (make_dataset.py sets it from collect.yaml)
EMBODIMENT = os.environ.get("CAPSIM_EMBODIMENT") or DEFAULT_EMBODIMENT
PROFILE = load_profile(EMBODIMENT)
FPS = int(PROFILE.recording.fps)
VIDEO_H, VIDEO_W = (int(v) for v in PROFILE.recording.frame_hw)   # the saver stores smaller frames; upscaled like the official set
VCODEC = str(PROFILE.recording.vcodec)                            # av1 like the official dataset
VCODEC_FALLBACK = str(PROFILE.recording.vcodec_fallback)          # when the encoder is missing
CAM_MAP = [(str(cam), str(key)) for cam, key in PROFILE.cameras.lerobot_keys.items()]   # (hdf5 camera group, LeRobot video key)
IMG_KEYS = [k for _, k in CAM_MAP]
STATE_KEY, ACTION_KEY = "observation.state", "action"


def downstream_root() -> Path:
    """The trainer repo whose LeRobot writer helpers define the output format (env.sh: TRAINER_ROOT).

    This module lives at <trainer>/codegen/sim, so with TRAINER_ROOT unset the enclosing repo is used."""
    root_env = os.environ.get("TRAINER_ROOT")
    root = Path(root_env) if root_env else Path(__file__).resolve().parents[4]
    src = root / "src"
    if not (src / "agenticwam").is_dir():
        raise RuntimeError(f"TRAINER_ROOT does not contain src/agenticwam: {root} (see env.sh)")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def decode_robotwin_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """RoboTwin hdf5 JPEG bytes -> true-RGB uint8 (H, W, 3). The [..., ::-1] is the BGR->RGB swap."""
    return np.array(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))[..., ::-1]


def decode_upscale(jpeg_bytes: bytes, size=(VIDEO_W, VIDEO_H)) -> Image.Image:
    img = Image.fromarray(np.ascontiguousarray(decode_robotwin_jpeg(jpeg_bytes)))
    return img if img.size == tuple(size) else img.resize(tuple(size), Image.BILINEAR)


def read_hdf5_episode(path):
    """-> (state (T, state_dim) f32, action = next state (last repeated), {lerobot video key: [jpeg bytes]}, T)."""
    with h5py.File(path, "r") as f:
        state = np.asarray(f["joint_action/vector"][:], dtype=np.float32)
        T = state.shape[0]
        action = np.empty_like(state)
        action[:-1] = state[1:]
        action[-1] = state[-1]
        cams = {key: [bytes(x) for x in f[f"observation/{cam}/rgb"][:]] for cam, key in CAM_MAP}
    return state, action, cams, T


def encode_frames(img_dir: Path, vid_path: Path) -> None:
    downstream_root()
    from agenticwam.datasets.lerobot.lerobot.datasets.video_utils import encode_video_frames
    vid_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encode_video_frames(img_dir, vid_path, FPS, vcodec=VCODEC, overwrite=True)
    except Exception:
        encode_video_frames(img_dir, vid_path, FPS, vcodec=VCODEC_FALLBACK, overwrite=True)
