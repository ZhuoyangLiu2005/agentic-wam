"""AGENTICWAM-Cosmos RoboTwin deploy policy (2-expert: video + action).

Adapts AgenticWAM's deploy_policy for the Cosmos-2.5-2B backbone:
  * observation  -> MULTIVIEW obs video [1,3,nc,224,320] in [-1,1] (views on the time axis,
                    per-view VAE encode), matching configs/robotwin/data/robotwin_official.yaml.
  * instruction  -> standalone Cosmos-Reason1 (Qwen2.5-VL) prompt_embs [1,L,100352].
  * AgenticWAMCosmos.infer_action(obs, proprio, prompt_embs) -> 64-D unified delta-EEF action chunk.

The 14-D Aloha qpos <-> 64-D delta-EEF round-trip (cuRobo FK/IK) is added on top by
deploy_policy_eef64.install() (model-agnostic, reused verbatim from AgenticWAM).

Model dtype: fp32 with include_text_encoder=False (the validated infer_action path). The
8.3B text encoder is kept OUT of the model (separate bf16 module) so the DiT stays uniformly
fp32 (avoids the assign=True mixed-dtype load path in load_cosmos_video_checkpoint).

RoboTwin script/eval_policy.py contract: get_model(usr_args), eval(TASK_ENV, model, obs),
reset_model(model), encode_obs(obs).
"""
import atexit
import glob
import logging
import os
import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

def _agenticwam_deploy_dtype():
    """Model dtype for deploy, env-selectable. Default fp32 = previous behaviour.

    AGENTICWAM_DEPLOY_DTYPE=bf16 loads the model in bf16 (what deploy/launch_server.sh
    sets: half the load time, no measured inference cost); unset keeps fp32.
    """
    import os
    return torch.bfloat16 if os.environ.get("AGENTICWAM_DEPLOY_DTYPE", "fp32").lower() in ("bf16", "bfloat16") else torch.float32


from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # <repo> (resolved through RoboTwin's policy symlink)
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agenticwam.datasets.lerobot.processors.agenticwam_processor import AgenticWAMProcessor  # noqa: E402
from agenticwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from agenticwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from agenticwam.datasets.dataset_utils import (  # noqa: E402
    ResizeSmallestSideAspectPreserving, CenterCrop, Normalize)
from torchvision.transforms.functional import (  # noqa: E402
    resize as _tv_resize, center_crop as _tv_center_crop)

logger = logging.getLogger(__name__)

# Camera order MUST match training (configs/robotwin/data/robotwin_official.yaml shape_meta.images):
#   cam_high, cam_left_wrist, cam_right_wrist  ->  RoboTwin sim head/left/right cameras.
_SIM_CAMERA_KEYS = ["head_camera", "left_camera", "right_camera"]


# ── generic helpers (verbatim from AgenticWAM deploy_policy) ──────────────────────
def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}") from exc
        return relative.as_posix()
    if _is_none_like(sim_cfg_name):
        return "robotwin/sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(sim_cfg_path, sim_cfg_name, sim_task, paths_overrides=None) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")
    if not _is_none_like(paths_overrides):          # weight locations from the eval entry point
        overrides.extend(o for o in str(paths_overrides).split("|") if o)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    # BUGFIX 2026-09-18: robotwin/eval/run_task.py forwards only a fixed --key list to RoboTwin's
    # eval_policy.py, so `+model.*` hydra overrides given to the entrypoint NEVER reach this
    # re-composed cfg (they do appear in the dumped eval_config.yaml, which is misleading).
    # An eval launched this way once silently ran without the code-vision prefix, with
    # code_vision=none -> no ViT prefix and the trained merger tensors rejected
    # ("unexpected=6" in the load line).
    # PREFERRED FIX is to declare code_vision INLINE in the task config
    # (configs/*/task/*.yaml), which survives re-composition -- that is why the real-world path
    # (deploy/common/policy_runtime.py, which composes `train` with task=<task>) was never hit.
    # This env fallback stays as a net for ad-hoc evals whose task config predates that rule.
    import os as _os
    OmegaConf.set_struct(cfg.model, False)   # a composed eval cfg may not carry these keys at all
    for _env, _key in (("CODE_VISION", "code_vision"),):
        _v = _os.environ.get(_env)
        if _v and str(cfg.model.get(_key, "")) != _v:
            cfg.model[_key] = _v
            print(f"[deploy] model.{_key}={_v} (from ${_env})", flush=True)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError("`dataset_stats_path` is required (pass from eval entrypoint overrides).")
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


# ── standalone Cosmos-Reason1 text encoder (bf16, kept out of the fp32 model) ──
def _text_encoder_state(path):
    """Cosmos-Reason1 weights from `path`: the public nvidia/Cosmos-Reason1-7B directory, or a Cosmos video
    checkpoint that embeds it under `text_encoder.text_encoder.*`. Returns (state_dict, config source)."""
    from safetensors.torch import load_file
    sd = {}
    for s in sorted(glob.glob(os.path.join(path, "model-*-of-*.safetensors"))):
        sd.update(load_file(s))
    if not sd:
        raise FileNotFoundError(f"no model-*-of-*.safetensors under {path} (paths.text_encoder)")
    pfx = "text_encoder.text_encoder."
    if any(k.startswith(pfx) for k in sd):
        sd = {k[len(pfx):]: v for k, v in sd.items() if k.startswith(pfx)}
    has_cfg = os.path.isfile(os.path.join(path, "config.json")) and os.path.isfile(os.path.join(path, "tokenizer.json"))
    return sd, (path if has_cfg else "nvidia/Cosmos-Reason1-7B")


class _CosmosTextEncoder:
    """Instruction -> [1,L,100352] Cosmos-Reason1 embed. Mirrors
    scripts/precompute_cosmos_text_embeds.py (same chat template + hidden-state concat) so eval
    embeds are identical to the training cache. Weights, config and tokenizer come from the
    Cosmos-Reason1-7B directory (paths.text_encoder)."""

    _SYS = "You are a helpful assistant who will provide prompts to an image generator."

    def __init__(self, ckpt_dir: str, device: str, max_len: int = 128, dtype=torch.bfloat16):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoConfig, AutoProcessor
        from accelerate import init_empty_weights
        sd, src = _text_encoder_state(ckpt_dir)
        cfg = AutoConfig.from_pretrained(src)
        with init_empty_weights():
            qwen = Qwen2_5_VLForConditionalGeneration(cfg)
        self._cfg_src = src
        # The weights use the OLDER transformers param naming (`visual.*` / `model.*`); the installed
        # transformers nests them under `model.visual.*` / `model.language_model.*`. Without remapping, the
        # language model does NOT match, stays uninitialized, and the encoder emits an ALL-ZERO embedding
        # (which silently mis-conditions the whole model -> noise video + garbage action). Remap so it loads.
        param_names = set(dict(qwen.named_parameters())) | set(dict(qwen.named_buffers()))
        remap = {}
        for kk, v in sd.items():
            if kk in param_names:
                cand = kk
            elif kk.startswith("visual.") and ("model." + kk) in param_names:
                cand = "model." + kk
            elif kk.startswith("model.") and ("model.language_model." + kk[len("model."):]) in param_names:
                cand = "model.language_model." + kk[len("model."):]
            else:
                cand = kk
            remap[cand] = v
        n = qwen.load_state_dict(remap, strict=False, assign=True)
        matched = len(remap) - len(n.unexpected_keys)
        logger.info("text encoder loaded: %d keys, matched=%d missing=%d unexpected=%d",
                    len(remap), matched, len(n.missing_keys), len(n.unexpected_keys))
        if matched < 0.5 * len(remap):
            raise RuntimeError(f"Cosmos-Reason1 load matched only {matched}/{len(remap)} keys — "
                               "param-name remap is wrong; the embedding would be garbage/zero.")
        # any still-unmatched (meta) params -> zeros so .to() works (these are unused at inference)
        for mod in qwen.modules():
            for pn, p in list(mod._parameters.items()):
                if p is not None and p.is_meta:
                    mod._parameters[pn] = torch.nn.Parameter(torch.zeros(p.shape, dtype=torch.float32),
                                                             requires_grad=False)
            for bn, b in list(mod._buffers.items()):
                if b is not None and b.is_meta:
                    mod._buffers[bn] = torch.zeros(b.shape, dtype=torch.float32)
        self.qwen = qwen.to(device, dtype).eval()
        self.proc = AutoProcessor.from_pretrained("nvidia/Cosmos-Reason1-7B")
        self.device = device
        self.max_len = int(max_len)
        # Optional for a policy sharing one GPU with the perception services.
        # Embeddings are still computed on the policy GPU and cached there.
        self.offload = _parse_bool(os.environ.get("AGENTICWAM_TEXT_OFFLOAD", "0"))
        self._cache: Dict[str, torch.Tensor] = {}
        # fail fast: a broken load yields a ~zero embedding that silently mis-conditions everything
        _probe = float(self.encode("pick up the object").std())
        self._cache.clear()
        logger.info("Cosmos-Reason1 encoder self-test: embedding std=%.4f", _probe)
        if _probe < 1e-3:
            raise RuntimeError(f"Cosmos-Reason1 encoder produced a ~zero embedding (std={_probe:.2e}) "
                               "— language-model weights did not load.")

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        if prompt in self._cache:
            return self._cache[prompt]
        if self.offload:
            self.qwen.to(self.device)
        conv = [
            {"role": "system", "content": [{"type": "text", "text": self._SYS}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        ids = self.proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=False,
                                            max_length=self.max_len, truncation=True,
                                            padding="max_length")
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if ids.ndim == 2:
            ids = ids[0]
        ids = ids[:self.max_len].unsqueeze(0)
        hs = self.qwen(ids, output_hidden_states=True).hidden_states
        emb = torch.cat([(h - h.mean(-1, keepdim=True)) / (h.std(-1, keepdim=True) + 1e-8)
                         for h in hs[1:]], dim=-1).detach()          # [1,L,100352]
        self._cache[prompt] = emb
        if self.offload:
            self.qwen.to("cpu")
            if torch.device(self.device).type == "cuda":
                torch.cuda.empty_cache()
        return emb


# ── policy ────────────────────────────────────────────────────────────────────
class AgenticWAMCosmosRobotWinPolicy:
    def __init__(self, model_cfg, processor_cfg, checkpoint_path, dataset_stats_path,
                 device, action_horizon, replan_steps, num_inference_steps, sigma_shift,
                 seed, text_ckpt_dir, text_max_len, n_cameras, num_video_frames,
                 video_log_dir=None):
        # The checkpoint supplies every trained weight (dit.*, proprio_encoder.*, code_vision.*); the frozen
        # VAE comes from wan_vae_path. No Cosmos video checkpoint is read at inference.
        mc = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        mc.include_text_encoder = False
        mc.load_pretrain = False
        self.model = instantiate(mc, model_dtype=_agenticwam_deploy_dtype(), device=device)
        if _is_none_like(checkpoint_path) or not Path(str(checkpoint_path)).exists():
            logger.warning("checkpoint '%s' missing -> untrained weights "
                           "(smoke/validation mode; NOT a trained policy).", checkpoint_path)
        else:
            self.model.load_checkpoint(str(checkpoint_path))
        self.model = self.model.to(device=device, dtype=_agenticwam_deploy_dtype()).eval()  # bf16 throughout: matches DeepSpeed bf16 training, consistent (no bf16/fp32 code-expert mismatch) + fast (fp32 video expert was ~20x slower in the 128-step code-gen cascade)
        self.device = device
        self.n_cameras = int(n_cameras)

        self.processor: AgenticWAMProcessor = instantiate(processor_cfg).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(dataset_stats_path)))

        self.text = _CosmosTextEncoder(text_ckpt_dir, device, max_len=text_max_len)

        # EXACT training obs transforms (video_size=[224,320]) -> pixel-parity with the data pipeline.
        self._resize = ResizeSmallestSideAspectPreserving(args={"img_w": 320, "img_h": 224})
        self._crop = CenterCrop(args={"img_w": 320, "img_h": 224})
        self._norm = Normalize(args={"mean": 0.5, "std": 0.5})
        # EVAL PREPROCESSING: mirror the training full-frame transform before resizing.
        # The three transforms above replicate only STAGE 2 of the training pipeline
        # (robot_video_dataset.py: video_size=[224,320] resize -> centre crop -> [-1,1] norm).
        # STAGE 1 -- the processor's train_transforms on the FULL frame -- was never applied at eval.
        # On the sim's native 240x320 D435 frame stage 2's resize is a no-op (scale =
        # max(320/320, 224/240) = 1.0) and the centre crop then drops 8 rows top and bottom, so eval
        # saw objects ~7% taller than training and lost content training always kept.
        #
        # AGENTICWAM_EVAL_PREPROC selects stage 1.  DEFAULT = "parity" (the fix is ON as of 2026-09-17).
        #   parity      : resize the full frame to [224,320]. Correct for every model whose
        #                 train_transforms are ToTensor+Resize (all shipped RoboTwin configs).
        #   parity_crop : centre-crop AGENTICWAM_EVAL_PREPROC_CROP (default 0.95) of the INPUT, then resize.
        #                 Correct for a model trained with RandomCrop(0.95)+Resize (the
        #                 train-RandomCrop / eval-CenterCrop pairing). The crop MUST be computed
        #                 from the actual input size -- a hardcoded 456x608 silently ZERO-PADS a
        #                 240x320 sim frame instead of erroring.
        #   legacy      : the pre-2026-09-17 behaviour, for reproducing published numbers exactly.
        _mode = (os.environ.get("AGENTICWAM_EVAL_PREPROC", "").strip().lower() or "parity")
        if _mode not in ("legacy", "parity", "parity_crop"):
            raise ValueError(f"AGENTICWAM_EVAL_PREPROC must be legacy|parity|parity_crop, got {_mode!r}")
        self._preproc_mode = _mode
        self._preproc_crop = float(os.environ.get("AGENTICWAM_EVAL_PREPROC_CROP", "0.95"))
        logger.info("[preproc] mode=%s%s", self._preproc_mode,
                    f" crop={self._preproc_crop}" if self._preproc_mode == "parity_crop" else "")

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.num_video_frames = int(num_video_frames)   # pixel frames/view for the WAM video rollout
        self.sigma_shift = sigma_shift
        self.code_cache = {}
        self.seed = seed
        # save the model's IMAGINED future video (WAM rollout) for visualization, beside the sim mp4s
        self.video_log_dir = Path(video_log_dir) if not _is_none_like(video_log_dir) else None
        if self.video_log_dir is not None:
            self.video_log_dir.mkdir(parents=True, exist_ok=True)
        self._save_all_replans = _parse_bool(os.environ.get("AGENTICWAM_SAVE_ALL_REPLANS", "0"))

        # [hybrid] AGENTICWAM_HYBRID=1: step-level code/WAM arbitration. Runtime package: robotwin/hybrid_eval/
        # (arbiter.py, assemble.py, config.py; launch.py / vision.py CLIs; docs/ROBOTWIN.md §6). _last_code hands each
        # replan's generated slice to the arbiter.
        self._last_code: Optional[str] = None
        self.hybrid = None
        if os.environ.get("AGENTICWAM_HYBRID") == "1":
            # RoboTwin loads this file through a symlinked policy dir, so resolve the real location
            # (realpath) to find ../hybrid_eval; AGENTICWAM_HYBRID_PKG (exported by hybrid_eval/launch.py) wins.
            _pkg_parent = os.environ.get("AGENTICWAM_HYBRID_PKG") or os.path.normpath(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
            if _pkg_parent not in sys.path:
                sys.path.insert(0, _pkg_parent)
            from hybrid_eval import arbiter as hybrid_arbiter
            self.hybrid = hybrid_arbiter.HybridArbiter(self.video_log_dir)
            logger.info("HYBRID arbitration active (code-primary; perception backend=%s)",
                        os.environ.get("AGENTICWAM_HYBRID_PERCEPTION") == "1")

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = -1   # reset() -> 0-based, aligns with RoboTwin's test_num
        # Per-episode success bookkeeping (see _flush_episode_results). `_ep_env` is the
        # TASK_ENV stashed by eval(); reset_model() never receives it.
        self._ep_env = None
        self._ep_logged = 0       # episodes already written to episodeN_result.json
        self._ep_suc_seen = 0     # TASK_ENV.suc as of the last flush
        self.step_count = 0
        logger.info("AgenticWAMCosmosRobotWinPolicy ready | ckpt=%s | nc=%d | horizon=%d | replan=%d | nis=%d",
                    checkpoint_path, self.n_cameras, self.action_horizon, self.replan_steps,
                    self.num_inference_steps)

    # ── state / action (34-D raw dual-EEF <-> 64-D via the training processor) ──
    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        meta = self.processor.shape_meta["state"]
        if len(meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        key = meta[0]["key"]
        batch = {"state": {key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action [B,T,D], got {tuple(action.shape)}")
        meta = self.processor.shape_meta["action"]
        key = meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][key]
        return normalizer.backward(action.to(dtype=torch.float32, device="cpu")).numpy()

    # ── multiview obs [1,3,nc,224,320] in [-1,1] ──────────────────────────────
    def _build_obs_video(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs = observation["observation"]
        views = []
        for cam in _SIM_CAMERA_KEYS[: self.n_cameras]:
            rgb = np.asarray(obs[cam]["rgb"]).astype(np.uint8)               # [H0,W0,3]
            views.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)  # [3,H0,W0] in [0,1]
        video = torch.stack(views, dim=0)                                    # [nc,3,H0,W0] (time=views)
        if self._preproc_mode != "legacy":
            # STAGE 1 = processor.train_transforms, reproduced on the raw sim frame.
            if self._preproc_mode == "parity_crop":
                _h, _w = video.shape[-2], video.shape[-1]
                video = _tv_center_crop(video, [int(round(self._preproc_crop * _h)),
                                                int(round(self._preproc_crop * _w))])
            video = _tv_resize(video, [224, 320], antialias=True)            # [nc,3,224,320]
        # STAGE 2 = robot_video_dataset video_size=[224,320]. Once stage 1 has run the resize/crop
        # are no-ops (224x320 in) and only the normalisation bites -- exactly as in training.
        video = self._resize(video)
        video = self._crop(video)
        video = self._norm(video)                                           # [nc,3,224,320] in [-1,1]
        video = video.permute(1, 0, 2, 3).unsqueeze(0)                      # [1,3,nc,224,320]
        return video.to(device=self.device, dtype=torch.float32)

    _SKILL_MAP = None

    def _skill_prompt(self) -> str:
        """EXP-1 retrieval skill block for the current task; '' when disabled."""
        path = os.environ.get("AGENTICWAM_TASK_SKILL_PROMPT")
        if not path:
            return ""
        if AgenticWAMCosmosRobotWinPolicy._SKILL_MAP is None:
            try:
                with open(path) as fh:
                    AgenticWAMCosmosRobotWinPolicy._SKILL_MAP = json.load(fh)
            except Exception:
                AgenticWAMCosmosRobotWinPolicy._SKILL_MAP = {}
        task = os.environ.get("TASK") or ""
        return AgenticWAMCosmosRobotWinPolicy._SKILL_MAP.get(task, "")

    _SLICE_TOTALS = None

    def _slice_list(self, task: str):
        """GT slice list for the oracle-history probe (same cache the trainer used)."""
        import hashlib, json as _json
        if getattr(AgenticWAMCosmosRobotWinPolicy, "_SLICE_LISTS", None) is None:
            AgenticWAMCosmosRobotWinPolicy._SLICE_LISTS = {}
        c = AgenticWAMCosmosRobotWinPolicy._SLICE_LISTS
        if task not in c:
            d = os.environ.get("AGENTICWAM_SLICE_CACHE_DIR", "")
            out = []
            if d:
                p = os.path.join(d, hashlib.sha256(task.encode("utf-8")).hexdigest() + ".json")
                try:
                    out = _json.load(open(p)).get("slices", [])
                except Exception:
                    out = []
            c[task] = out
        return c[task]

    def _slice_total(self, task: str) -> int:
        """Cycle count n for the '# Completed: k/n' log -- read from the SAME slice cache training
        used (keyed sha256(task)); cached across calls; defaults 1 when the task has no slices."""
        import hashlib, json as _json
        if AgenticWAMCosmosRobotWinPolicy._SLICE_TOTALS is None:
            AgenticWAMCosmosRobotWinPolicy._SLICE_TOTALS = {}
        t = AgenticWAMCosmosRobotWinPolicy._SLICE_TOTALS
        if task not in t:
            n = 1
            d = os.environ.get("AGENTICWAM_SLICE_CACHE_DIR", "")
            if d:
                p = os.path.join(d, hashlib.sha256(task.encode("utf-8")).hexdigest() + ".json")
                try:
                    n = max(1, len(_json.load(open(p)).get("slices", [])))
                except Exception:
                    n = 1
            t[task] = n
        return t[task]

    _CANON_MAP = None

    def _canonical_instruction(self, instruction: str) -> str:
        """Replace the sim's randomised instruction with the canonical one the model TRAINED on.
        Enabled only when AGENTICWAM_TASK_INSTRUCTION_MAP (json {task_name: instruction}) is set and the
        task name is resolvable from the TASK / EVALUATION task_name env."""
        path = os.environ.get("AGENTICWAM_TASK_INSTRUCTION_MAP")
        if not path:
            return instruction
        if AgenticWAMCosmosRobotWinPolicy._CANON_MAP is None:
            try:
                with open(path) as fh:
                    AgenticWAMCosmosRobotWinPolicy._CANON_MAP = json.load(fh)
            except Exception:
                AgenticWAMCosmosRobotWinPolicy._CANON_MAP = {}
        task = os.environ.get("TASK") or os.environ.get("EVAL_TASK_NAME") or ""
        return AgenticWAMCosmosRobotWinPolicy._CANON_MAP.get(task, instruction)

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        obs_video = self._build_obs_video(observation)
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state).to(self.device)              # [1,64]
        # Training set `episode_instruction_map`, which replaces each episode's randomised
        # instruction with ONE canonical primary per task (robot_video_dataset.py:285-290).
        # The sim returns a random paraphrase (click_bell has 91 distinct strings), so ~94% of
        # eval episodes were OOD for the code expert. Env-gated: unset -> unchanged behaviour.
        # DUAL INSTRUCTION (epi_* arms): prompt_embs uses the CANONICAL task instruction (that is what
        # the video/action experts were trained on, and why the embed cache stays small), while the CODE
        # expert gets THIS episode's own instruction. Keep the raw one before the canonical override.
        episode_instruction = instruction
        instruction = self._canonical_instruction(instruction)
        wrapped_instruction = DEFAULT_PROMPT.format(task=instruction)
        # EXP-1: prefix the SAME retrieval skill block the code expert was TRAINED with
        # (dataset skill_prompt_map). Env-gated: unset -> unchanged behaviour.
        # SLICE MODE (EXP-2): training builds the code prompt from the RAW instruction with a
        # completed-slice history block -- "# Instruction: %s\n# Completed:\n%s# Next:\n"
        # (robot_video_dataset.py::_get_code_slice), NOT the DEFAULT_PROMPT-wrapped form. Sending the
        # wrapped form there would repeat exactly the mismatch that made the code task-blind.
        if os.environ.get("AGENTICWAM_SKILL_PROMPT_ONLY"):
            # skill_prompt_only=true runs (robot_video_dataset.py:321): the code expert
            # was trained on ONLY the skill block, no instruction appended. Sending the
            # wrapped instruction too would be the same OOD-prompt mismatch as before.
            code_instruction = self._skill_prompt()
        elif os.environ.get("AGENTICWAM_SLICE_SUMMARY"):
            # MEMFULL deploy: skill block + BOUNDED event log, byte-identical to the training
            # composer (robot_video_dataset._get_code_slice, code_slices_summary=True). k = slices
            # generated so far this episode (self-history, matching selfgen training), clamped to
            # n-1 (training never saw k=n); n = the task's GT cycle count from the SAME slice cache.
            _sl = getattr(self, "_exec_slices", [])
            _n = self._slice_total(instruction)
            _k = min(len(_sl), max(0, _n - 1))
            if os.environ.get("AGENTICWAM_SLICE_FAKE_K"):
                _k = 0            # frozen counter: episode progress never advances in the prompt
            _last = _sl[-1] if _sl else ""
            if os.environ.get("AGENTICWAM_SLICE_ORACLE_HIST"):
                _gt = self._slice_list(instruction)
                _last = _gt[_k - 1] if (_k > 0 and _k - 1 < len(_gt)) else ""
            if _last and not _last.endswith("\n"):
                _last += "\n"
            code_instruction = self._skill_prompt() + (
                "# Completed: %d/%d cycles. Last:\n%s# Next:\n" % (_k, _n, _last))
        elif os.environ.get("AGENTICWAM_SLICE_MODE"):
            _hist = getattr(self, "_mem_code_history", "") or ""
            code_instruction = ("# Instruction: %s\n# Completed:\n%s# Next:\n" % (instruction, _hist))
        elif os.environ.get("AGENTICWAM_CODE_PREAMBLE_FILE"):
            # Stage-slice training uses the raw task after '# Instruction:'.
            # The model adds the helper-library/next-stage framing; only the
            # video/action text encoder consumes DEFAULT_PROMPT's wrapper.
            code_instruction = (episode_instruction if os.environ.get("AGENTICWAM_CODE_EPISODE_INSTR")
                                else instruction)
        else:
            _base = (DEFAULT_PROMPT.format(task=episode_instruction)
                     if os.environ.get("AGENTICWAM_CODE_EPISODE_INSTR") else wrapped_instruction)
            code_instruction = self._skill_prompt() + _base
        prompt_embs = self.text.encode(wrapped_instruction).to(self.device)
        want_video = self.video_log_dir is not None and (self._save_all_replans or self.step_count == 0)
        use_code = bool(getattr(self.model, "use_code_expert", False))
        with torch.no_grad():
            if use_code:
                out = self.model.infer_action_trimodal(
                    obs_video=obs_video, proprio=proprio, prompt_embs=prompt_embs,
                    action_horizon=self.action_horizon, num_video_frames=self.num_video_frames,
                    num_inference_steps=self.num_inference_steps, sigma_shift=self.sigma_shift,
                    # Composed above to match this checkpoint's training prompt:
                    # skill + wrapped instruction, or raw task for stage slices.
                    instruction=code_instruction, code_cache=self.code_cache, return_video=want_video,
                )
            else:
                out = self.model.infer_action(
                    obs_video=obs_video, proprio=proprio, prompt_embs=prompt_embs,
                    action_horizon=self.action_horizon, num_video_frames=self.num_video_frames,
                    num_inference_steps=self.num_inference_steps, sigma_shift=self.sigma_shift,
                    return_video=want_video,
                )
        gen_video = None
        self._last_code = None                                              # [hybrid] slice for the arbiter
        if use_code:
            act = out[0]                                                    # (act, code[, gen_video])
            if len(out) >= 2:
                self._last_code = str(out[1])
            if len(out) >= 2 and self.video_log_dir is not None:
                # [hybrid] episode{N}_codes.jsonl: EVERY replan's generated code beside the videos.
                try:
                    import json as _cj
                    from pathlib import Path as _CP
                    _cd = _CP(self.video_log_dir); _cd.mkdir(parents=True, exist_ok=True)
                    with open(_cd / ("episode%d_codes.jsonl" % self.episode_count), "a") as _cf:
                        _cf.write(_cj.dumps({"replan": int(self.step_count), "code": str(out[1])}) + "\n")
                except Exception as _ce:
                    logger.warning("codes.jsonl append failed: %s", _ce)
            if os.environ.get("AGENTICWAM_SLICE_SUMMARY") and len(out) >= 2 and isinstance(out[1], str):
                # MEMFULL self-history: the slice just generated becomes the next replan's
                # "Last:" entry and extends the slots-encoder text (model._mem_code_history).
                _g = out[1].strip()
                if _g:
                    if not hasattr(self, "_exec_slices"):
                        self._exec_slices = []
                    self._exec_slices.append(_g)
                    _h = getattr(self.model, "_mem_code_history", "") or ""
                    self.model._mem_code_history = _h + _g + "\n"
            if self.step_count == 0 and len(out) >= 2:
                print("[ep %d] POLICY_CODE (real obs):\n%s\n[/POLICY_CODE]" % (self.episode_count, out[1]), flush=True)
                try:
                    self._save_code(out[1])
                except Exception as _e:
                    logger.warning('save code failed: %s', _e)
            if want_video and len(out) >= 3:
                gen_video = out[2]
        elif want_video:
            act, gen_video = out
        else:
            act = out                                                       # [T,64] fp32 cpu
        if gen_video is not None:
            try:
                self._save_generated_video(gen_video, instruction)
            except Exception as e:
                logger.warning("save generated video failed (ep %s step %s): %s",
                               self.episode_count, self.step_count, e)
        if os.environ.get("AGENTICWAM_DEMO") and self.video_log_dir is not None and use_code and len(out) >= 2:
            try:                                                    # per-replan demo capture (code + obs RGB)
                _d = self.video_log_dir
                (_d / ("episode%d_step%d_code.txt" % (self.episode_count, self.step_count))).write_text(str(out[1]))
                import imageio as _iio, numpy as _np
                _rgb = _np.asarray(observation["observation"]["head_camera"]["rgb"]).astype("uint8")
                _iio.imwrite(str(_d / ("episode%d_step%d_obs.jpg" % (self.episode_count, self.step_count))), _rgb)
            except Exception as _e:
                logger.warning("demo capture failed: %s", _e)
        return self._denormalize_action(act)[0]                             # [T,64]

    def _save_code(self, code):
        """Save the episode's generated policy code to a .txt beside the gen video for inspection."""
        if self.video_log_dir is None:
            return
        from pathlib import Path as _P
        d = _P(self.video_log_dir); d.mkdir(parents=True, exist_ok=True)
        (d / ("episode%d_code.txt" % self.episode_count)).write_text(str(code))

    def _save_generated_video(self, gen_video, instruction):
        """Save the model's IMAGINED future video ([1,3,nc*Tpix,H,W] in [-1,1]) as an mp4 with the
        nc views tiled horizontally per frame, beside RoboTwin's rollout mp4s in the eval output dir."""
        import imageio
        v = gen_video[0]                                     # [3, nc*Tpix, H, W]
        C, T, H, W = v.shape
        nc = self.n_cameras
        Tpv = max(1, T // nc)
        v = v[:, :nc * Tpv].reshape(C, nc, Tpv, H, W)
        frames = []
        for t in range(Tpv):
            row = torch.cat([v[:, view, t] for view in range(nc)], dim=-1)   # [3,H,nc*W]
            img = (((row.clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255)
                   .byte().permute(1, 2, 0).cpu().numpy())                   # [H,nc*W,3]
            frames.append(img)
        out = self.video_log_dir / f"episode{self.episode_count}_step{self.step_count}_gen.mp4"
        imageio.mimsave(str(out), frames, fps=4)

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        chunk = self._infer_action_chunk(observation, instruction)
        for i in range(min(self.replan_steps, chunk.shape[0])):
            self.pending_actions.append(np.asarray(chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError("Observation required at replan step (empty action queue).")
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)
            if self.hybrid is not None and self._last_code is not None:
                # [hybrid] step-level arbitration: if the code channel consumed this
                # replan, drop the freshly generated WAM chunk and re-observe.
                handled = self.hybrid.on_replan(task_env, self.episode_count,
                                                self.step_count, self._last_code)
                if handled:
                    self.pending_actions.clear()
                    self.step_count += self.replan_steps
                    return
        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return
        action = self.pending_actions.popleft()
        task_env.take_action(action, action_type="qpos")
        self.step_count += 1

    def _flush_episode_results(self) -> None:
        """Write per-episode success to episodeN_result.json, from RoboTwin's counters.

        Why this exists: with --no-render there is no episodeN_*success-*.mp4, and the
        arbiter's `check_success` is only recorded on replans that executed code, so it is
        absent on whole episodes. That left run_meta.json's log-scraped "Success rate: k/n"
        as the only source, which is shard-granular (a task moves in steps of 7) and loses
        everything if a shard dies mid-way.

        RoboTwin's eval_policy.py advances BOTH `TASK_ENV.suc` and `TASK_ENV.test_num` at the
        end of an episode, before the next episode calls reset_model(). So at flush time they
        describe the episode that just finished, and that episode's success is the increase in
        `suc` since the previous flush. The last episode has no following reset, so get_model()
        also registers this as an atexit hook.

        One file per episode (overwrite, not append): re-running a shard into a directory that
        already has results cannot leave stale or doubled records.
        """
        env = self._ep_env
        if env is None or self.video_log_dir is None:
            return
        try:
            n_done = int(getattr(env, "test_num", 0))
            cum_suc = int(getattr(env, "suc", 0))
        except Exception:
            return
        n_new = n_done - self._ep_logged
        if n_new <= 0:
            return
        d_suc = cum_suc - self._ep_suc_seen
        task = getattr(env, "task_name", None)
        for i in range(n_new):
            idx = self._ep_logged + i
            rec = {"episode": idx,
                   # n_new > 1 means a flush was missed, so the delta cannot be attributed to
                   # a single episode; record it as unknown rather than guess.
                   "success": bool(d_suc) if n_new == 1 else None,
                   "cum_suc": cum_suc, "cum_total": n_done,
                   "ts": datetime.now().isoformat(timespec="seconds")}
            if task:
                rec["task"] = str(task)
            if n_new != 1:
                rec["ambiguous"] = True
            try:
                (self.video_log_dir / ("episode%d_result.json" % idx)).write_text(
                    json.dumps(rec) + "\n")
            except Exception as e:  # noqa: BLE001
                logger.warning("episode%d_result.json write failed: %s", idx, e)
        self._ep_logged = n_done
        self._ep_suc_seen = cum_suc

    def reset(self) -> None:
        self.pending_actions.clear()
        self.episode_count += 1
        if self.hybrid is not None:
            self.hybrid.on_reset()          # [hybrid] per-episode arbiter state
        try:
            self.model._mem_code_history = ""   # E3: reset code memory per episode
            self._exec_slices = []              # MEMFULL: reset the event-log self-history too
        except Exception:
            pass
        self.step_count = 0
        self.code_cache = {}          # per-episode: regenerate code each episode


# ── RoboTwin entrypoint contract ──────────────────────────────────────────────
def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    cfg = _compose_sim_cfg(usr_args.get("sim_cfg_path"), usr_args.get("sim_cfg_name"),
                           usr_args.get("sim_task"), usr_args.get("sim_paths_overrides"))
    OmegaConf.set_struct(cfg, False)
    cfg.model.use_code_expert = _parse_bool(usr_args.get("use_code_expert", False))
    cfg.model.code_model_name = usr_args.get("code_model_name") or cfg.model.get("code_model_name", "mock")

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` (checkpoint path) is required.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to cpu.")
        device = "cpu"

    dataset_stats_path = _resolve_dataset_stats_path(usr_args.get("dataset_stats_path"))

    # [dataprep] dataprep-generated data configs nest the per-dataset settings
    # (DeadlineDataset -> ConcatDataset -> dataset blocks); the flat v2-style
    # configs carry num_frames/processor directly on data.train. Resolve once.
    _dtrain = cfg.data.train
    if "num_frames" not in _dtrain:
        _dtrain = _dtrain.dataset.datasets[0]

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eh = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eh if eh is not None else int(_dtrain.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", 20))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_ckpt = str(cfg.paths.text_encoder)
    text_max_len = int(usr_args.get("text_max_len", os.environ.get("AGENTICWAM_TEXT_MAX_LEN", 128)))
    n_cameras = int(cfg.model.get("n_cameras", 3))
    # pixel frames/view of the WAM video rollout (obs + future), matches the training horizon
    _nf = int(_dtrain.num_frames)
    _fr = int(_dtrain.get("action_video_freq_ratio", 4))
    num_video_frames = (_nf - 1) // _fr + 1

    policy = AgenticWAMCosmosRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=_dtrain.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_ckpt_dir=text_ckpt,
        text_max_len=text_max_len,
        n_cameras=n_cameras,
        num_video_frames=num_video_frames,
        video_log_dir=(None if _is_none_like(usr_args.get("eval_output_dir"))
                       else str(usr_args.get("eval_output_dir"))),
    )
    # cuRobo warmup: initialize the cuRobo CUDA context so the RoboTwin sim's CuroboPlanner
    # (set_planner->right_planner) inits reliably; the eef64 path did this implicitly via its IK import.
    try:
        from curobo.types.base import TensorDeviceType as _TDTwarm
        _TDTwarm()
    except Exception as _we:
        logger.warning('cuRobo warmup skipped: %s', _we)
    if str(usr_args.get("action_space", os.environ.get("AGENTICWAM_ACTION_SPACE", ""))).lower() == "eef64":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import deploy_policy_eef64
        deploy_policy_eef64.install(policy)
        logger.info("EEF64 adapter installed (action_space=eef64)")
    # The final episode has no following reset_model(), so flush it at interpreter exit.
    atexit.register(policy._flush_episode_results)
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    # reset_model(model) is called without TASK_ENV, so stash it here for
    # _flush_episode_results (RoboTwin calls eval() at least once per episode).
    model._ep_env = TASK_ENV
    model.step(TASK_ENV, encode_obs(observation))


def reset_model(model):
    model._flush_episode_results()   # record the episode that just finished
    model.reset()
