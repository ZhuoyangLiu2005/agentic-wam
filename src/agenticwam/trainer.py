import logging
import json
import inspect
import os
import re
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta as _pg_timedelta
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class AgenticWAMTrainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.keep_last_ckpts = int(cfg.get('keep_last_ckpts', 0) or 0)  # rolling ckpts; 0=keep all
        # Persist the (large) DeepSpeed optimizer/grad state for resuming. Default OFF:
        # we don't resume, and it writes a ~90GB dir per checkpoint (fills disk).
        self.save_optimizer_state = bool(cfg.get("save_optimizer_state", False))
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.foundation_ckpt = cfg.get("foundation_ckpt", None)
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        # Startup straggler tolerance. accelerate's InitProcessGroupKwargs only covers the DEFAULT
        # process group (PG 0); DeepSpeed creates a SECOND group (PG 1) that falls back to
        # ProcessGroupNCCL's 600s C++ default. A 48-rank pretraining run died there: each rank ran a
        # ~26 min dataset init with >10 min spread, so the first big ZeRO ALLREDUCE
        # (NumelIn=996,156,416) blew the 600s watchdog. Raise BOTH, and log what actually took.
        _pg_min = int(os.environ.get("AGENTICWAM_PG_TIMEOUT_MIN", "120"))
        _pg_to = _pg_timedelta(minutes=_pg_min)
        # NOTE: for the NCCL backend torch resolves NEW process-group timeouts via
        # _get_default_timeout(), which returns the module-global `default_pg_nccl_timeout`
        # (10 min, from the C++ _DEFAULT_PG_NCCL_TIMEOUT) -- NOT `default_pg_timeout` (30 min).
        # Patching only the latter silently does nothing; the log line below proved that.
        try:
            import torch.distributed.constants as _dc
            import torch.distributed.distributed_c10d as _c10d
            for _m in (_dc, _c10d):
                _m.default_pg_timeout = _pg_to
                if hasattr(_m, "default_pg_nccl_timeout"):
                    _m.default_pg_nccl_timeout = _pg_to
        except Exception:                                                # noqa: BLE001
            pass
        os.environ.setdefault("DEEPSPEED_TIMEOUT", str(_pg_min))

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[InitProcessGroupKwargs(timeout=_pg_to)],
        )
        try:
            import torch.distributed as _dist
            if _dist.is_initialized():
                logger.info("process-group timeout: requested=%s min | default_pg=%s",
                            _pg_min, _dist.distributed_c10d._get_default_timeout(_dist.get_backend()))
        except Exception:                                                # noqa: BLE001
            pass
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        # `model.dit` includes the optional code expert; respect its frozen Qwen body.
        trainable_params = [p for p in self.model.dit.parameters() if p.requires_grad]
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(list(proprio_encoder.parameters()))
        # FIX 2026-09-03: side modules outside model.dit (E3 code memory, ViT merger) were never in the
        # optimizer -> frozen at init for every run to date.
        for _name, _m in getattr(self.model, "extra_trainable_modules", lambda: [])():
            _ps = [p for p in _m.parameters() if p.requires_grad]
            trainable_params.extend(_ps)
            logger.info("extra trainable module %s: %d tensors / %.2fM params", _name, len(_ps),
                        sum(p.numel() for p in _ps) / 1e6)
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        # FIX (resume-clobber, 2026-07-16): a weights-only .pt resume MUST be loaded into
        # the RAW model BEFORE accelerator.prepare(). Under DeepSpeed ZeRO bf16, prepare
        # snapshots fp32 master weights from the current params, so a post-prepare
        # load_state_dict is silently reverted on the first optimizer.step().
        self._loaded_weights_pre_prepare = self._maybe_load_weights_before_prepare()

        # FIX 2026-09-07 (grad-accum): accelerate only fills DeepSpeed keys that are present as "auto" in the
        # JSON; scripts/ds_configs/*.json have no "gradient_clipping", so DeepSpeed ran with clip=0 and
        # cfg.max_grad_norm was silently ignored (accelerator.clip_grad_norm_ is only a norm GETTER under
        # DeepSpeed). Observed: clip=0 in DeepSpeed, grad norms 15-160 fed to AdamW unclipped.
        _dsp = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if _dsp is not None:
            _dsp.deepspeed_config["gradient_clipping"] = float(self.max_grad_norm)
        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        # Deep prefetch + persistent workers keep a large batch buffer warm so the
        # dataloader does not drain every ~40 steps and stall ~150s on Lustre refill
        # (the periodic multi-node data-starvation stalls). Only valid with workers>0.
        loader_perf_kwargs = {}
        if self.num_workers > 0:
            loader_perf_kwargs["persistent_workers"] = True
            loader_perf_kwargs["prefetch_factor"] = 4
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
            **loader_perf_kwargs,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _maybe_load_weights_before_prepare(self) -> bool:
        """Load a weights-only .pt resume into the raw model BEFORE accelerator.prepare().
        Returns True if it loaded (so the post-prepare path skips it). Directory
        (full training-state) resume is still handled after prepare."""
        fnd = getattr(self, "foundation_ckpt", None)
        if fnd:
            fp = Path(str(fnd))
            if not fp.exists():
                raise FileNotFoundError(f"foundation_ckpt not found: {fnd}")
            logger.info("Foundation-init (shape-filtered transfer) BEFORE prepare: %s", fnd)
            self.model.load_foundation(str(fp))
            return True
        resume = self.resume
        if not resume:
            return False
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            return False
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint BEFORE prepare (DeepSpeed-safe): %s", resume)
        self.model.load_checkpoint(str(resume_path), optimizer=None)
        return True

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if getattr(self, "_loaded_weights_pre_prepare", False):
            logger.info("Weight checkpoint already loaded before prepare; skipping post-prepare load.")
            self._restore_step_from_weights_name(resume_path)
            self._restore_dataloader_progress_weights_only(resume_path)
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer moments were not restored under ZeRO2.")
        self._restore_step_from_weights_name(resume_path)
        self._restore_dataloader_progress_weights_only(resume_path)

    def _restore_step_from_weights_name(self, resume_path):
        """A weights-only .pt carries NO step / LR-schedule state -- those live in the ZeRO shards,
        which cannot be resharded across world sizes (48 -> 64 GPUs). Without this, moving cluster
        silently restarts the LR schedule at warmup while the WEIGHTS are 20k steps trained, which
        both mis-schedules the LR and desynchronises this arm from the others in the ablation.
        Recover the step from the filename (step_020000.pt) and fast-forward the scheduler.
        Optimizer MOMENTS are still lost -- they are the sharded part and are unrecoverable here.
        Override with AGENTICWAM_RESUME_STEP=<n>; set 0 to disable."""
        import re as _re
        env = os.environ.get("AGENTICWAM_RESUME_STEP")
        if env is not None:
            n = int(env)
        else:
            m = _re.search(r"step[_-](\d+)", Path(str(resume_path)).name)
            n = int(m.group(1)) if m else 0
        if n <= 0:
            return
        self.global_step = int(n)
        adv = 0
        try:
            for _ in range(int(n)):
                self.scheduler.step(); adv += 1
        except Exception as e:                                     # noqa: BLE001
            logger.warning("LR fast-forward stopped at %d/%d: %s", adv, n, e)
        logger.info("Weights-only resume: restored global_step=%d, LR fast-forwarded to %.3e "
                    "(optimizer moments NOT restored)", self.global_step,
                    float(self.optimizer.param_groups[0]["lr"]))
        # LR RE-WARMUP (AGENTICWAM_RESUME_REWARMUP=<steps>, default off). With the Adam moments gone the
        # first updates are ~lr*sign(g) for EVERY param; ramp lr linearly from lr/N to the scheduled
        # lr over N optimizer steps. The cosine scheduler is chainable (recurses on group['lr']), so
        # the train loop restores the TRUE scheduled lr right before each scheduler.step() and
        # re-applies the ramp after it -- the schedule itself is never perturbed.
        _n = int(os.environ.get("AGENTICWAM_RESUME_REWARMUP", "0") or 0)
        self._rewarm_n = max(0, _n); self._rewarm_k = 0
        if self._rewarm_n > 0:
            self._rewarm_true = [float(g["lr"]) for g in self.optimizer.param_groups]
            _f0 = 1.0 / self._rewarm_n
            for g, t in zip(self.optimizer.param_groups, self._rewarm_true):
                g["lr"] = t * _f0
            logger.info("LR re-warmup ACTIVE: %d steps, lr %.3e -> %.3e", self._rewarm_n,
                        self._rewarm_true[0] * _f0, self._rewarm_true[0])

    def _restore_dataloader_progress_weights_only(self, resume_path):
        """A weights-only resume used to restart the sampler at batch 0 of the epoch (same seed ->
        same permutation -> replays already-seen windows). The dataloader position lives in the
        state dir's trainer_state.json, which is independent of the world-size/param-list-bound
        ZeRO shards, so restore it exactly as load_training_state does. Source:
        AGENTICWAM_RESUME_TRAINER_STATE=<file|dir>, else <run>/checkpoints/state/step_{step:06d}/."""
        src = os.environ.get("AGENTICWAM_RESUME_TRAINER_STATE")
        cand = None
        if src:
            p = Path(src)
            cand = (p / "trainer_state.json") if p.is_dir() else p
        elif self.global_step > 0:
            cand = Path(self.state_dir) / f"step_{self.global_step:06d}" / "trainer_state.json"
        if cand is None or not cand.exists():
            logger.warning("Weights-only resume: no trainer_state.json (%s); dataloader restarts at batch 0", cand)
            return
        with open(cand, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "epoch" not in payload or "batch_in_epoch" not in payload:
            logger.warning("Weights-only resume: %s lacks epoch/batch_in_epoch; dataloader restarts at batch 0", cand)
            return
        self.epoch = int(payload["epoch"])
        self.batch_in_epoch = int(payload["batch_in_epoch"])
        self.train_sampler.set_epoch_offset(self.epoch)
        self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
        logger.info("Weights-only resume: restored dataloader progress from %s: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    cand, self.epoch, self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes)

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
        code_memory = getattr(model, "code_memory", None)
        if code_memory is not None:
            code_memory.train()
            code_memory.requires_grad_(True)
        # The code expert lives inside model.dit; the blanket unfreeze above would
        # un-freeze its pretrained Qwen body. Restore its freeze policy.
        if hasattr(model, "apply_param_freeze_policy"):
            model.apply_param_freeze_policy()
        # FIX 2026-09-03: re-enable the trainable side modules that live OUTSIDE model.dit
        # (E3 code memory; ViT merger) -- the blanket freeze above silently froze them.
        for _name, _m in getattr(model, "extra_trainable_modules", lambda: [])():
            _m.train()
            _m.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        prompt_embs = sample.get("prompt_embs", None)
        if isinstance(prompt_embs, torch.Tensor) and prompt_embs.ndim == 2:
            prompt_embs = prompt_embs.unsqueeze(0)
        policy_code = sample.get("policy_code", None)
        if isinstance(policy_code, str):
            policy_code = [policy_code]
        action_is_pad = sample.get("action_is_pad", None)
        if isinstance(action_is_pad, torch.Tensor) and action_is_pad.ndim == 1:
            action_is_pad = action_is_pad.unsqueeze(0)
        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
            "prompt_embs": prompt_embs,
            "policy_code": policy_code,
            "action_is_pad": action_is_pad,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, val_loss_dict = model.training_loss(sample)
            val_loss = val_loss.float().item()
        # Held-out loss (video/action/code) is the val metric. Gather scalar means across ranks.
        keys = sorted((val_loss_dict or {}).keys())
        def _fv(x):
            return float(x.mean().item()) if hasattr(x, "mean") else float(x)
        vals = [float(val_loss)] + [_fv(val_loss_dict[k]) for k in keys]
        local = torch.tensor(vals, device=self.accelerator.device, dtype=torch.float32).unsqueeze(0)
        gathered = self.accelerator.gather_for_metrics(local)
        mean = gathered.mean(dim=0)
        if was_dit_training:
            self._set_dit_only_train_mode()
        result = {"val_loss": float(mean[0].item())}
        for _i, _k in enumerate(keys):
            result["val_" + str(_k)] = float(mean[_i + 1].item())
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
            if self.keep_last_ckpts > 0:
                import glob as _glob
                _cks = sorted(_glob.glob(os.path.join(self.weights_dir, 'step_*.pt')))
                for _old in _cks[:-self.keep_last_ckpts]:
                    try:
                        os.remove(_old)
                    except OSError:
                        pass
        self.accelerator.wait_for_everyone()

        state_path = None
        if self.save_optimizer_state:
            state_path = os.path.join(self.state_dir, step_tag)
            ensure_dir(state_path)
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self.accelerator.wait_for_everyone()

        if self.save_optimizer_state and self.accelerator.is_main_process and self.keep_last_ckpts > 0:
            import glob as _glob2, shutil as _shutil
            _sts = sorted(_glob2.glob(os.path.join(self.state_dir, 'step_*')))
            for _old in _sts[:-self.keep_last_ckpts]:
                _shutil.rmtree(_old, ignore_errors=True)
        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        # ---- THROUGHPUT-DECAY INSTRUMENTATION (AGENTICWAM_PHASE_TIMING=0 to disable) ----
        # Throughput halves over ~14h of RUN AGE and fully resets on restart, identically on two
        # independent clusters. Externally excluded: thermal/clocks, GPU-memory growth, data
        # starvation (0 DATA_WAIT/205 samples), cgroup CPU throttling, IB->TCP fallback, and
        # time-of-day load. The p25 step time doubles while p90 stays flat, and GPUs report >90%
        # util at 120W/700W (spinning in collectives). Remaining candidates are only visible
        # in-process: allocator fragmentation (num_alloc_retries / reserved-vs-allocated) and
        # per-phase collective time. This logs both every log_every steps.
        # SAMPLED, not every micro-step: a synchronize() after fwd+bwd on EVERY micro-step
        # serialises the pipeline and measurably slows training (first attempt: 244s/20steps vs
        # ~110s uninstrumented). Measure only the micro-steps belonging to ONE optimizer step per
        # log window (~1 in 80), so the overhead is negligible while still tracking the trend.
        _ph_on = os.environ.get("AGENTICWAM_PHASE_TIMING", "1") != "0"
        _ph = {"data": 0.0, "skew": 0.0, "fwd": 0.0, "bwd": 0.0, "opt": 0.0, "n": 0}
        _ph_win = max(int(self.log_every), 1)
        def _measuring():
            return _ph_on and (self.global_step % _ph_win == 0)
        def _sync():
            if _measuring() and torch.cuda.is_available():
                torch.cuda.synchronize()

        while self.global_step < self.max_steps:
            try:
                _t0 = time.perf_counter()
                sample = next(data_iter)
                _ph["data"] += time.perf_counter() - _t0
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                # SKEW vs COMPUTE: `fwd` is timed with a synchronize() AFTER forward, so in a
                # distributed job any inter-rank skew carried in from the previous step lands in
                # whichever phase syncs first -- making `fwd` look like it grows when the real
                # cause may be ranks drifting apart. Barrier + sync FIRST: the wait is charged to
                # `skew`, and the `fwd` that follows is measured from an aligned, drained state,
                # so it is pure forward compute. Measured steps only (~1 in 80), and the barrier
                # is entered on the same global_step by every rank, so it cannot deadlock.
                if _measuring():
                    _t0 = time.perf_counter()
                    try:
                        import torch.distributed as _dist
                        if _dist.is_available() and _dist.is_initialized():
                            _dist.barrier()
                    except Exception:
                        pass
                    _sync()
                    _ph["skew"] = _ph.get("skew", 0.0) + time.perf_counter() - _t0

                _t0 = time.perf_counter()
                with self.accelerator.autocast():
                    # FIX 2026-09-07 (grad-accum): call the WRAPPER (DeepSpeedEngine.forward -> AgenticWAMCosmos.forward ->
                    # training_loss), not `.training_loss` on the module. DeepSpeed 0.18 scales the loss gradient
                    # by 1/gradient_accumulation_steps via hooks it registers on the outputs of engine.forward();
                    # bypassing forward() skipped that, so with gas>1 the accumulated gradient was the SUM of the
                    # micro-batch gradients (gas x too large; the reported grad_norm was inflated the same way).
                    # Under DDP the same bypass would also skip the gradient-sync hooks.
                    loss, loss_dict = train_model(sample)
                _sync()
                if _measuring(): _ph["fwd"] += time.perf_counter() - _t0

                _t0 = time.perf_counter()
                self.accelerator.backward(loss)
                _sync()
                if _measuring():
                    _ph["bwd"] += time.perf_counter() - _t0
                    _ph["n"] += 1

                if self.accelerator.sync_gradients:
                    _t0 = time.perf_counter()
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        _rw = getattr(self, "_rewarm_n", 0)
                        if _rw > 0:   # restore the TRUE scheduled lr: the cosine recursion must not see the ramp
                            for g, t in zip(self.optimizer.param_groups, self._rewarm_true):
                                g["lr"] = t
                        self.scheduler.step()
                        if _rw > 0:
                            self._rewarm_true = [float(g["lr"]) for g in self.optimizer.param_groups]
                            self._rewarm_k += 1
                            _f = min(1.0, (self._rewarm_k + 1) / float(_rw))
                            for g, t in zip(self.optimizer.param_groups, self._rewarm_true):
                                g["lr"] = t * _f
                            if self._rewarm_k + 1 >= _rw:
                                self._rewarm_n = 0
                                logger.info("LR re-warmup complete at step %d (lr=%.3e)", self.global_step + 1, self._rewarm_true[0])
                    self.optimizer.zero_grad(set_to_none=True)
                    _sync()
                    if _measuring(): _ph["opt"] += time.perf_counter() - _t0; _ph["nopt"] = _ph.get("nopt",0)+1
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        _n = max(_ph["n"], 1)
                        _ms = torch.cuda.memory_stats() if torch.cuda.is_available() else {}
                        _res = _ms.get("reserved_bytes.all.current", 0) / 1e9
                        _alloc = _ms.get("allocated_bytes.all.current", 0) / 1e9
                        _retries = _ms.get("num_alloc_retries", 0)
                        _frag = (1.0 - _alloc / _res) if _res > 0 else 0.0
                        _age_h = (time.perf_counter() - self.run_start_time) / 3600.0
                        logger.info(
                            description
                            + " | age=%.1fh phase(ms/microstep) data=%.0f skew=%.0f fwd=%.0f bwd=%.0f opt=%.0f"
                              " | alloc_retries=%d reserved=%.1fGB allocated=%.1fGB frag=%.2f"
                            % (_age_h, 1e3*_ph["data"]/_n, 1e3*_ph.get("skew",0.0)/_n, 1e3*_ph["fwd"]/_n, 1e3*_ph["bwd"]/_n,
                               1e3*_ph["opt"]/max(_ph.get("nopt",0),1),
                               _retries, _res, _alloc, _frag))

                        wandb_payload = {
                            "perf/phase_data_ms": 1e3*_ph["data"]/_n,
                            "perf/phase_skew_ms": 1e3*_ph.get("skew",0.0)/_n,
                            "perf/phase_fwd_ms": 1e3*_ph["fwd"]/_n,
                            "perf/phase_bwd_ms": 1e3*_ph["bwd"]/_n,
                            "perf/phase_opt_ms": 1e3*_ph["opt"]/max(_ph.get("nopt",0),1),
                            "perf/alloc_retries": _retries,
                            "perf/reserved_gb": _res,
                            "perf/frag": _frag,
                            "perf/run_age_h": _age_h,
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)
                        _ph = {"data": 0.0, "skew": 0.0, "fwd": 0.0, "bwd": 0.0, "opt": 0.0, "n": 0, "nopt": 0}

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            _comp = " ".join("%s=%.4f" % (k, v) for k, v in metrics.items() if k.startswith("val_") and k != "val_loss")
                            logger.info("[eval] step=%d val_loss=%.4f | %s", self.global_step, metrics["val_loss"], _comp)
                            eval_payload = {"eval/val_loss": float(metrics["val_loss"])}
                            for k, v in metrics.items():
                                if k.startswith("val_") and k != "val_loss":
                                    eval_payload["eval/" + k] = float(v)
                            self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
