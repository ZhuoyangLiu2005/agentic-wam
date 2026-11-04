"""XPolicyLab adapter: serve an AgenticWAM checkpoint post-trained on RoboDojo (dual ARX X5) to the RoboDojo sim.

Lives in agentic-wam at deploy/robodojo/AgenticWAM and is symlinked to RoboDojo/XPolicyLab/policy/AgenticWAM by
deploy/robodojo/install.sh. The model runs through deploy/common/policy_runtime.py with --embodiment x5
(deploy/common/x5_eef64.py: X5 FK in, IK out), so the sim sees absolute joint targets (action_type=joint).

Observation -> model, exactly as utils/convert_robodojo_v21_to_yam82.py built the training data:
    cam_head / cam_left_wrist / cam_right_wrist 480x640 RGB -> rows 16:464 -> area-resize 224x320
        -> head_camera / left_camera / right_camera (the training order cam_high, cam_left_wrist, cam_right_wrist)
    state [L arm 6, L gripper, R arm 6, R gripper] (pack_robot_state, joint)
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state

AGENTICWAM_ROOT = Path(__file__).resolve().parents[3]          # deploy/robodojo/AgenticWAM/model.py -> agentic-wam
SRC_H, SRC_W, CROP_Y0, CROP_Y1 = 480, 640, 16, 464
OUT_H, OUT_W = 224, 320
CAMERAS = (("cam_head", "head_camera"), ("cam_left_wrist", "left_camera"), ("cam_right_wrist", "right_camera"))


def _none_like(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in ("", "none", "null"))


def training_view(rgb: np.ndarray) -> np.ndarray:
    """480x640 sim frame -> the 224x320 training frame (crop rows 16:464, area resize)."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.shape[:2] != (SRC_H, SRC_W):
        raise ValueError(f"expected {SRC_H}x{SRC_W} camera frames (the RoboDojo training resolution), got {rgb.shape}")
    return cv2.resize(rgb[CROP_Y0:CROP_Y1], (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)


def _load_policy_runtime():
    """agentic-wam/deploy/common as package `agenticwam_deploy_common`: the XPolicyLab server already has a top-level
    `deploy` module (the policies' deploy.py), so `import deploy.common` cannot be used here."""
    import importlib.util
    name = "agenticwam_deploy_common"
    if name not in sys.modules:
        pkg_dir = AGENTICWAM_ROOT / "deploy" / "common"
        spec = importlib.util.spec_from_file_location(name, pkg_dir / "__init__.py",
                                                      submodule_search_locations=[str(pkg_dir)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return importlib.import_module(f"{name}.policy_runtime").PolicyRuntime


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        cfg = dict(model_cfg)
        self.cfg = cfg
        self.action_type = cfg["action_type"]
        if self.action_type != "joint":
            raise ValueError(f"AgenticWAM serves absolute joints (X5 IK on the server); use action_type=joint, "
                             f"got {self.action_type!r}")
        self.robot_action_dim_info = get_robot_action_dim_info(cfg["env_cfg_type"])
        self.instruction = str(cfg["instruction"])          # the one instruction the checkpoint was trained on
        self.replan_steps = int(cfg.get("replan_steps") or 24)
        self.verbose = bool(cfg.get("verbose", True))

        ckpt = cfg.get("checkpoint_path")
        dataset = cfg.get("dataset")
        if _none_like(ckpt) or not Path(str(ckpt)).is_file():
            raise FileNotFoundError(f"checkpoint_path={ckpt!r} is not a file")
        if _none_like(dataset) or not (Path(str(dataset)) / "dataset_stats_eef64.json").is_file():
            raise FileNotFoundError(f"dataset={dataset!r} must be the training dataset dir (dataset_stats_eef64.json)")

        # The serving defaults of deploy/launch_server.sh.
        for k, v in (("AGENTICWAM_DEPLOY_DTYPE", "bf16"), ("AGENTICWAM_FAST_JOINT", "1"), ("AGENTICWAM_CODE_PIN", "1"),
                     ("AGENTICWAM_NOISE_SEED", "1234"), ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"),
                     ("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface"),
                     ("TORCHINDUCTOR_CACHE_DIR", str(AGENTICWAM_ROOT / ".torchinductor_cache_deploy"))):
            os.environ.setdefault(k, v)
        PolicyRuntime = _load_policy_runtime()

        overrides = [f"model.infer_code_tokens={int(cfg.get('infer_code_tokens') or 128)}"]
        if not _none_like(cfg.get("paths")):
            overrides.append(f"paths={cfg['paths']}")
        self.rt = PolicyRuntime(
            ckpt=str(ckpt), task=str(cfg.get("task") or "posttrain_yam"), dataset=str(dataset), overrides=overrides,
            replan_steps=self.replan_steps, num_inference_steps=int(cfg.get("num_inference_steps") or 6),
            embodiment="x5", device="cuda")
        self.replan_steps = int(self.rt.replan_steps)
        print(f"[AgenticWAM] warmup {self.rt.warmup(height=OUT_H, width=OUT_W):.1f}s | ckpt={ckpt} | "
              f"replan={self.replan_steps} nis={self.rt.num_inference_steps} | instruction={self.instruction!r}",
              flush=True)
        self.last = None
        self.n_chunks = 0

    def _encode(self, obs):
        images = {dst: training_view(obs["vision"][src]["color"]) for src, dst in CAMERAS}
        state = pack_robot_state(obs, "joint", self.robot_action_dim_info, source_type="obs", state_type="state")
        return images, np.asarray(state, np.float64).reshape(-1)

    def update_obs(self, obs):
        self.last = self._encode(obs)

    def update_obs_batch(self, obs_list):
        self._batch = {int(o["env_idx"]): self._encode(o) for o in obs_list}

    def _act(self, enc):
        if enc is None:
            raise ValueError("call update_obs() before get_action()")
        images, state = enc
        chunk, code, infer_ms = self.rt.infer(images, state, self.instruction)
        chunk = np.asarray(chunk[: self.replan_steps], np.float32)
        chunk[:, 6] = np.clip(chunk[:, 6], 0.0, 1.0)
        chunk[:, 13] = np.clip(chunk[:, 13], 0.0, 1.0)
        self.n_chunks += 1
        if self.verbose:
            first = (code or "").strip().splitlines()[:1]
            print(f"[AgenticWAM] chunk {self.n_chunks}: {infer_ms:.0f} ms, ik_failures(total)={self.rt.ik_failures}"
                  + (f" | code: {first[0][:100]}" if first else ""), flush=True)
        return unpack_robot_state(chunk, "joint", self.robot_action_dim_info, source_type="obs")

    def get_action(self):
        return self._act(self.last)

    def get_action_batch(self, env_idx_list):
        return [self._act(self._batch[int(i)]) for i in env_idx_list]

    def reset(self):
        self.last = None
        self.n_chunks = 0
        self.rt.ik_failures = 0
        self.rt.reset()
