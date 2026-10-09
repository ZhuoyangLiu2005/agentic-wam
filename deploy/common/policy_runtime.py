#!/usr/bin/env python
"""Policy runtime — build the model once, turn an observation into an ABSOLUTE qpos chunk.

Transport-free on purpose: no HTTP, no sockets, no robot I/O. Both deploy lines (local GPU-on-robot,
remote GPU-on-cluster) drive this same object, so the thing that actually produces motion is
identical in both and can't drift between them.

The build and inference path are lifted verbatim from the verified `deploy/deploy_server.py`:
hydra-compose the task config -> AgenticWAMCosmosRobotWinPolicy -> _infer_action_chunk -> add the request's
state back (RelativeJointTransform.backward). Nothing here changes what the model sees.

Three things it adds over the inline version in deploy_server.py:

  * code capture     — deploy_server.py hardcodes `"code": None` because _infer_action_chunk drops
                       the generated program (it is only printed at step 0). We wrap
                       model.infer_action_trimodal to keep it, so the plan is visible every replan.
  * honest dtype     — reports the dtype actually loaded, read off the parameters
                       (AGENTICWAM_DEPLOY_DTYPE: fp32 by default, bf16 via deploy/launch_server.sh). /health should not repeat a claim
                       it hasn't checked, since the whole latency budget hangs off it.
  * warmup           — the first inference is 3-5x slower (CUDA init). Burning it at startup means the
                       robot never eats that stall on a live observation.
"""
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# deploy/common/policy_runtime.py -> parents[2] == agenticwam_cosmos root
AGENTICWAM = str(Path(__file__).resolve().parents[2])

logger = logging.getLogger(__name__)


class PolicyRuntime:
    """Owns the model. Thread-safe: one GPU, so inference is serialized under a lock."""

    def __init__(self, ckpt: str, task: str = "posttrain_yam", dataset: Optional[str] = None, overrides: Optional[list] = None,
                 dataset_stats: Optional[str] = None,
                 action_horizon: Optional[int] = None, replan_steps: int = 24,
                 num_inference_steps: int = 10, device: str = "cuda",
                 video_log_dir: Optional[str] = None, text_max_len: int = 128):
        sys.path.insert(0, AGENTICWAM + "/src")
        sys.path.insert(0, AGENTICWAM + "/robotwin/deploy")
        os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
        os.chdir(AGENTICWAM)  # hydra composes with project-relative paths

        import torch
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        import deploy_policy as dp

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=os.path.join(AGENTICWAM, "configs")):
            cfg = compose(config_name="train", overrides=[f"task={task}"] + ([f"data.dataset_root={dataset}"] if dataset else [])
                          + list(overrides or []))
        OmegaConf.set_struct(cfg, False)
        self.cfg = cfg

        self.dataset_stats = dataset_stats or str(cfg.data.train.pretrained_norm_stats)
        nf = int(cfg.data.train.num_frames)
        fr = int(cfg.data.train.get("action_video_freq_ratio", 4))
        self.action_horizon = int(action_horizon) if action_horizon else (nf - 1)
        self.num_video_frames = (nf - 1) // fr + 1
        self.n_cameras = int(cfg.model.get("n_cameras", 3))
        self.model_action_dim = int(cfg.model.get("action_dim", 14))
        # 64-D unified-EEF checkpoints are served as 14-D joints: FK on the way
        # in, IK on the way out (yam_eef64). The client-facing action_dim is
        # therefore 14 in that mode -- deploy/robot/agenticwam_client.py refuses
        # anything else, and the request's state is validated against it.
        self.unified_eef = self.model_action_dim == 64
        if self.unified_eef:
            transforms = [str(t.get("_target_", "")) for t in
                          cfg.data.train.processor.get("action_state_transforms", None) or []]
            if not any(t.endswith("YamUnifiedEEFTransform") for t in transforms):
                raise NotImplementedError(
                    f"task {task!r} uses {transforms}: the 64-D <-> joint bridge (yam_eef64) is YAM's "
                    "kinematics only, so this checkpoint cannot be served (docs/REALWORLD.md §4.8)")
            from . import yam_eef64
            self._eef64 = yam_eef64
            self.action_dim = 14
        else:
            self.action_dim = self.model_action_dim
        self.ik_failures = 0
        self.n_rtc_guided = 0
        self._episode_code_ids = None   # AGENTICWAM_CODE_PIN: step-0 program, frozen per episode
        self.task = task
        self.ckpt = ckpt

        ckpt_exists = Path(str(ckpt)).exists()
        if not ckpt_exists:
            logger.warning("checkpoint %r does not exist -> untrained weights. "
                           "This is SMOKE MODE: shapes flow but the actions are meaningless.", ckpt)

        t0 = time.perf_counter()
        self.policy = dp.AgenticWAMCosmosRobotWinPolicy(
            model_cfg=cfg.model,
            processor_cfg=cfg.data.train.processor,
            checkpoint_path=ckpt,
            dataset_stats_path=self.dataset_stats,
            device=device,
            action_horizon=self.action_horizon,
            replan_steps=replan_steps,
            num_inference_steps=num_inference_steps,
            sigma_shift=None,
            seed=None,
            text_ckpt_dir=str(cfg.paths.text_encoder),
            text_max_len=text_max_len,
            n_cameras=self.n_cameras,
            num_video_frames=self.num_video_frames,
            video_log_dir=video_log_dir,
        )
        self.build_sec = time.perf_counter() - t0
        self.smoke_mode = not ckpt_exists
        self.use_code = bool(getattr(self.policy.model, "use_code_expert", False))
        self.replan_steps = int(self.policy.replan_steps)
        self.num_inference_steps = int(self.policy.num_inference_steps)
        self.device = device

        # dtype as LOADED, not as documented (see module docstring)
        try:
            self.model_dtype = str(next(self.policy.model.parameters()).dtype)
        except StopIteration:
            self.model_dtype = "unknown"

        # Env-gated torch.compile of the MoT core -- the per-denoise-step (and
        # per-code-block) forward that dominates latency. Modes (reduce-overhead == CUDA graphs): "reduce-overhead" | "default" | "max-autotune";
        # unset -> eager, unchanged behaviour. First inference per token-shape
        # pays the compile; the startup warmup absorbs the warmup-shape one,
        # and the first real /act pays one more (real prompt = new code length).
        # Compiled-denoise-step override (deploy/common/fast_joint_infer.py):
        # compiles the WHOLE per-step core, not just the MoT. Supersedes
        # AGENTICWAM_TORCH_COMPILE for joint-objective checkpoints; do not combine.
        self.fast_joint = os.environ.get("AGENTICWAM_FAST_JOINT", "") not in ("", "0")
        if self.fast_joint:
            from . import fast_joint_infer
            fast_joint_infer.install(self.policy.model)

        self.compile_mode = "" if self.fast_joint else os.environ.get("AGENTICWAM_TORCH_COMPILE", "")
        if self.compile_mode:
            # fullgraph verified to hold (profile_cudagraph_test: no graph breaks).
            # Measured +6% end to end -- the MoT's launches shrink, but most of the
            # per-step eager machinery (video_pre/pre_dit/schedulers/unmasking)
            # sits OUTSIDE this region and still dominates the CPU gap.
            self.policy.model.mot = torch.compile(
                self.policy.model.mot, mode=self.compile_mode, fullgraph=True)
            logger.info("torch.compile enabled on model.mot (mode=%s, fullgraph)", self.compile_mode)

        self._last_code: Optional[str] = None
        self._install_code_capture()

        self._lock = threading.Lock()
        self.warmed_up = False
        self.n_infer = 0
        self.policy.reset()

        logger.info("PolicyRuntime ready | task=%s | ckpt=%s | code_expert=%s | dtype=%s | "
                    "horizon=%d replan=%d nis=%d | build %.1fs",
                    task, ckpt, self.use_code, self.model_dtype, self.action_horizon,
                    self.replan_steps, self.num_inference_steps, self.build_sec)

    def _install_code_capture(self) -> None:
        """Keep the generated program instead of dropping it. Pure passthrough otherwise."""
        if not self.use_code:
            return
        model = self.policy.model
        original = model.infer_action_trimodal

        def _capture(*args, **kwargs):
            out = original(*args, **kwargs)
            try:
                self._last_code = out[1] if isinstance(out, tuple) and len(out) >= 2 else None
            except Exception:  # never let telemetry break inference
                self._last_code = None
            return out

        model.infer_action_trimodal = _capture

    def _rtc_stage(self, state: np.ndarray, committed, freeze) -> bool:
        """Convert committed 14-D joint rows into normalized 64-D targets +
        weights and stage them on the model for the fast-joint denoise loop
        (inference-only RTC: hard-freeze the delay prefix, soft-mask the rest).
        Returns True if staged. No-ops unless unified_eef + AGENTICWAM_FAST_JOINT."""
        import torch
        if not (self.unified_eef and self.fast_joint) or os.environ.get("AGENTICWAM_RTC") == "0":
            return False
        cj = np.asarray(committed, dtype=np.float64)
        if cj.ndim != 2 or cj.shape[1] != 14 or len(cj) == 0:
            return False
        cj = cj[:24]                                   # never freeze most of the horizon
        rows64 = self._eef64.joints_to_delta64(cj, state)
        key = self.policy.processor.shape_meta["action"][0]["key"]
        norm = self.policy.processor.normalizer.normalizers["action"][key]
        k = len(cj)
        # Committed rows are the first k rows of the new chunk; per-step action stats ([horizon, 64]) are
        # sliced to those rows (norm.forward would broadcast a [k, 64] block against all `horizon` rows).
        scale, offset = norm.scale, norm.offset
        if scale.dim() == 2:
            scale, offset = scale[:k], offset[:k]
        targets = torch.clamp(torch.as_tensor(rows64, dtype=torch.float32) * scale + offset, -5.0, 5.0)
        f = int(max(0, min(int(freeze or 0), k)))
        w = np.zeros(k)
        w[:f] = 1.0
        if k > f:                                      # RTC soft mask: decay to 0
            w[f:] = np.linspace(0.8, 0.0, k - f, endpoint=False)
        self.policy.model._rtc_targets = targets
        self.policy.model._rtc_weights = torch.as_tensor(w, dtype=torch.float32)
        return True

    # ── inference ──────────────────────────────────────────────────────────────
    def infer(self, images: Dict[str, np.ndarray], state: np.ndarray,
              instruction: str, committed=None, freeze=None,
              ) -> Tuple[np.ndarray, Optional[str], float]:
        """{camera: HxWx3 uint8}, raw joint state -> (ABSOLUTE qpos chunk [T,D], code, infer_ms).

        The model predicts deltas; we resolve them against the caller's state here, which is why
        the response is safe to command directly as joint targets. Joint-space checkpoints
        (RelativeJointTransform) add the state back; unified-EEF checkpoints go through FK/IK
        (yam_eef64), so the client sees 14-D absolute joints either way.
        """
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        model_state = self._eef64.state14_to_82(state) if self.unified_eef else state
        observation = {
            "observation": {cam: {"rgb": images[cam]} for cam in images},
            "joint_action": {"vector": model_state.tolist()},
        }
        with self._lock:
            self._last_code = None
            rtc_on = False
            if committed is not None:
                try:
                    rtc_on = self._rtc_stage(state, committed, freeze)
                except Exception as exc:  # noqa: BLE001 -- RTC is an assist, never a blocker
                    logger.warning("RTC staging failed (%s); running unguided", exc)
            if os.environ.get("AGENTICWAM_CODE_PIN") and self.fast_joint:
                self.policy.model._pinned_code_ids = self._episode_code_ids
            t0 = time.perf_counter()
            try:
                rel = np.asarray(self.policy._infer_action_chunk(observation, instruction),
                                 dtype=np.float64)
            finally:
                # staged guidance is strictly per-request; a leftover would
                # silently guide the NEXT request toward stale actions
                self.policy.model._rtc_targets = None
                self.policy.model._rtc_weights = None
            if rtc_on:
                self.n_rtc_guided += 1
            if (os.environ.get("AGENTICWAM_CODE_PIN") and self.fast_joint
                    and self._episode_code_ids is None):
                canvas = getattr(self.policy.model, "_last_code_canvas", None)
                if canvas is not None:
                    self._episode_code_ids = canvas
                    logger.info("AGENTICWAM_CODE_PIN: episode program frozen (%d tokens)",
                                int(canvas.shape[0]))
            if self.unified_eef:
                chunk, fails = self._eef64.decode_chunk(rel, state)
                if fails:
                    self.ik_failures += fails
                    logger.warning("IK missed tolerance on %d arm-rows this chunk "
                                   "(episode total %d) -- unreachable pose requested?",
                                   fails, self.ik_failures)
            else:
                chunk = rel + state[None, :]
            infer_ms = (time.perf_counter() - t0) * 1e3
            self.policy.step_count += 1
            self.n_infer += 1
            code = self._last_code
        return chunk, code, infer_ms

    def warmup(self, height: int = 480, width: int = 640) -> float:
        """Burn the first-call CUDA cost on synthetic input. Returns seconds spent."""
        from .contract import CAMERAS
        blank = np.zeros((height, width, 3), np.uint8)
        t0 = time.perf_counter()
        try:
            self.infer({c: blank for c in CAMERAS}, np.zeros(self.action_dim), "warmup")
        finally:
            with self._lock:                      # warmup must not count as episode progress
                self.policy.step_count = 0
                self.n_infer = 0
                self._episode_code_ids = None     # nor pin its (garbage-input) program
        self.warmed_up = True
        return time.perf_counter() - t0

    def reset(self) -> int:
        """New episode: clears the per-episode code cache and the pending-action queue."""
        with self._lock:
            self.policy.reset()
            self._episode_code_ids = None
            if self.fast_joint:
                self.policy.model._pinned_code_ids = None
        return int(self.policy.episode_count)

    def info(self) -> Dict[str, Any]:
        return {"task": self.task, "ckpt": self.ckpt, "smoke_mode": self.smoke_mode,
                "code_expert": self.use_code, "action_dim": self.action_dim,
                "model_action_dim": self.model_action_dim,
                "action_space": "eef64" if self.unified_eef else "joint",
                "ik_failures": self.ik_failures,
                "torch_compile": self.compile_mode or "off",
                "fast_joint": self.fast_joint,
                "n_rtc_guided": self.n_rtc_guided,
                "code_pin": bool(os.environ.get("AGENTICWAM_CODE_PIN")),
                "code_pinned_now": self._episode_code_ids is not None,
                "action_horizon": self.action_horizon, "replan_steps": self.replan_steps,
                "num_inference_steps": self.num_inference_steps, "n_cameras": self.n_cameras,
                "model_dtype": self.model_dtype, "device": self.device,
                "dataset_stats": self.dataset_stats, "warmed_up": self.warmed_up,
                "episode": int(self.policy.episode_count), "n_infer": self.n_infer,
                "action_type": "qpos_abs", "build_sec": round(self.build_sec, 1)}
