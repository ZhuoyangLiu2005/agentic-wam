"""AgenticWAM top-level tri-modal model.

Cosmos-2.5-2B video expert + Wan-style action/code experts, coupled by `GenericMoT`
(per-layer joint self-attention).

Scheduler pairing (each expert keeps the scheduler it was designed/pretrained with):
  * video  -> Cosmos `FlowMatchUniPCScheduler` (the video pretraining: sigma convention,
              velocity target = noise - x0, conditioning frames get sigma = cond_timestep).
  * action -> AgenticWAM `WanContinuousFlowMatchScheduler` (matched pair with `ActionDiT`;
              `sample_training_t` / 3-arg `training_target`).

Video path follows the Cosmos video pretraining (patchify + cond-mask -> 3D RoPE -> embedded_timestep
-> MultiView cross-attn to Reason1 text -> norm_out -> unpatchify). Action/code follow AgenticWAM.

Weight loading: `load_cosmos_video_checkpoint` remaps the checkpoint's `transformer.*`/`vae.*`/
`text_encoder.*`/`original_video_tokenizer.*` into this module key-for-key.

VERIFY-at-smoke tags mark the few inferred details (per-token sigma expansion, multiview
cond-frame layout, the pretraining's exact `encode_video` preprocessing).
"""
from __future__ import annotations

import os

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from agenticwam.utils.logging_config import get_logger

_VIZ = {"active": False, "done": False, "steps": []}
from .generic_mot import GenericMoT
from .cosmos_video_expert import CosmosVideoExpert
from .wan_experts import WanActionExpert, WanCodeExpert

logger = get_logger(__name__)


class _DiTCore(nn.Module):
    """The trainable transformer core (`model.dit`): the MoT experts + the video tokenizer +
    the (parameter-free) multiview RoPE. Grouping these under one submodule lets the trainer
    freeze-except-dit, optimize `dit.parameters()`, and checkpoint only the trainable weights
    (the frozen VAE / Cosmos-Reason1 stay outside and are reconstructed at load)."""

    def __init__(self, mot: nn.Module, video_tokenizer: nn.Module, video_rope: nn.Module):
        super().__init__()
        self.mot = mot
        self.video_tokenizer = video_tokenizer
        self.video_rope = video_rope


class AgenticWAMCosmos(nn.Module):
    def __init__(
        self,
        video_expert: CosmosVideoExpert,
        action_expert: WanActionExpert,
        code_expert: Optional[WanCodeExpert],
        vae: nn.Module,
        video_tokenizer: nn.Module,
        video_rope: nn.Module,
        text_encoder: nn.Module,
        video_scheduler: Any,
        action_scheduler: Any,
        *,
        proprio_dim: int = 0,
        text_proj_dim: int = 1024,
        n_cameras: int = 4,
        cond_timestep: float = 0.1,
        num_cond_frames: int = 0,   # # observation pixel frames/view kept clean (sigma=cond_timestep); rest generated
        video_sigma_shift: float = 5.0,
        video_noise_beta: Optional[tuple] = None,   # (alpha,beta,noise_s) e.g. (7.0,1.0,0.999); None=uniform
        tiled: bool = False,
        tile_size: tuple = (34, 34),
        tile_stride: tuple = (18, 16),
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_code: float = 0.5,
        mot_checkpoint_mixed_attn: bool = True,
        use_code_expert: bool = True,
        infer_code_tokens: Optional[int] = None,
    ):
        super().__init__()
        # INFERENCE canvas length for the code stream, decoupled from the TRAINING cap.
        # They are not the same quantity: build_code_batch's max_code_tokens budgets
        # (prompt + code) TOGETHER, while inference's gen_len is the code-only all-<MASK>
        # canvas. Reusing the training cap at inference makes the canvas 5-25x longer than
        # real targets, which is out of distribution (training canvases are the batch-max
        # true length) and makes the model pad with repeated filler -- measured: code-GT
        # parse 0/48 at canvas 512 vs 27/48 at canvas 64. None -> fall back to the cap.
        self.infer_code_tokens = None if infer_code_tokens is None else int(infer_code_tokens)
        experts: Dict[str, nn.Module] = {"video": video_expert, "action": action_expert}
        if use_code_expert and code_expert is not None:
            # code in the MIDDLE — the video->code->action pipeline (aligned with AgenticWAM's
            # cascaded design): the video expert is the autonomous world model, the code expert
            # plans from the predicted future video, and the action executes conditioned on the code.
            experts = {"video": video_expert, "code": code_expert, "action": action_expert}
        mot = GenericMoT(experts, checkpoint_mixed_attn=mot_checkpoint_mixed_attn)
        self.use_code_expert = use_code_expert and code_expert is not None

        # Trainable transformer core. The trainer optimizes/checkpoints `model.dit`; keeping the
        # frozen VAE/text-encoder OUT of it makes freeze-except-dit correct and keeps ckpts small.
        self.dit = _DiTCore(mot, video_tokenizer, video_rope)
        self.vae = vae
        self.text_encoder = text_encoder
        self.video_scheduler = video_scheduler
        # FlowMatchUniPC.set_timesteps() READS+OVERWRITES self.sigmas, so each inference call
        # re-shifts the schedule from the previous call's output (compounding). Snapshot the
        # pristine TRAINING schedule and restore it before every set_timesteps (see infer_action),
        # otherwise the schedule degenerates over an episode -> black video + frozen arm.
        self._vs_sigmas0 = video_scheduler.sigmas.detach().clone()
        self._vs_timesteps0 = video_scheduler.timesteps.detach().clone()
        self.action_scheduler = action_scheduler
        self.proprio_encoder = nn.Linear(proprio_dim, text_proj_dim) if proprio_dim > 0 else None

        self.n_cameras = n_cameras
        self.cond_timestep = cond_timestep
        self.num_cond_frames = num_cond_frames
        self.video_sigma_shift = video_sigma_shift
        self.video_noise_beta = video_noise_beta
        self.tiled = tiled
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.loss_lambda_video = loss_lambda_video
        self.loss_lambda_action = loss_lambda_action
        self.loss_lambda_code = loss_lambda_code

        for p in self.vae.parameters():
            p.requires_grad_(False)
        if self.text_encoder is not None:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

    # ── accessors into the trainable core (properties → no double param registration) ──
    @property
    def mot(self):
        return self.dit.mot

    @property
    def video_tokenizer(self):
        return self.dit.video_tokenizer

    @property
    def video_rope(self):
        return self.dit.video_rope

    @property
    def video_expert(self):
        return self.dit.mot.experts["video"]

    @property
    def action_expert(self):
        return self.dit.mot.experts["action"]

    @property
    def code_expert(self):
        if self.use_code_expert and "code" in self.dit.mot.experts:
            return self.dit.mot.experts["code"]
        return None

    @property
    def device(self):
        return next(self.parameters()).device

    # ── trainer interface: freeze policy + weights-only checkpoints ──────────────
    def apply_param_freeze_policy(self):
        """Called by the trainer after the blanket dit-unfreeze. Re-freeze the frozen
        backbones and the code expert's pretrained Qwen body."""
        for m in (self.vae, self.text_encoder):
            if m is None:
                continue
            m.eval()
            m.requires_grad_(False)
        ce = self.code_expert
        if ce is not None and hasattr(ce, "set_backbone_frozen"):
            ce.set_backbone_frozen(frozen=True)

    def extra_trainable_modules(self):
        """Top-level trainable side modules that live OUTSIDE self.dit: the E3 learned code memory
        and the Qwen3-VL merger of the code-vision prefix. FIX 2026-09-03: the trainer's
        freeze-except-dit, its optimizer param list and the weights-only checkpoint all enumerated
        `dit` (+proprio) only, so these were silently frozen at random/HF init AND never saved
        (see docs/HANDOFF_memory_vit_freeze_fix.md). Returns [(state_dict prefix, module)].
        AGENTICWAM_FREEZE_VIT_MERGER=1 keeps the merger frozen (arm-comparability with pre-fix ViT runs)."""
        out = []
        cm = getattr(self, "code_memory", None)
        if cm is not None:
            out.append(("code_memory", cm))
        cv = getattr(self, "code_vision", None)
        mg = getattr(getattr(cv, "vit", None), "merger", None) if cv is not None else None
        if mg is not None and not os.environ.get("AGENTICWAM_FREEZE_VIT_MERGER"):
            out.append(("code_vision.vit.merger", mg))
        return out

    def _trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {f"dit.{k}": v for k, v in self.dit.state_dict().items()}
        if self.proprio_encoder is not None:
            for k, v in self.proprio_encoder.state_dict().items():
                sd[f"proprio_encoder.{k}"] = v
        for name, m in self.extra_trainable_modules():      # FIX 2026-09-03: memory + ViT merger are saved
            for k, v in m.state_dict().items():
                sd[f"{name}.{k}"] = v
        return sd

    @torch.no_grad()
    def save_checkpoint(self, path: str, optimizer=None, step: Optional[int] = None):
        torch.save({"model": self._trainable_state_dict(), "step": step}, path)
        return path

    @torch.no_grad()
    def load_checkpoint(self, path: str, optimizer=None):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        res = self.load_state_dict(sd, strict=False)  # frozen backbones intentionally absent
        logger.info(f"agenticwam load_checkpoint {path}: loaded {len(sd)} trainable tensors "
                    f"(unexpected={len(res.unexpected_keys)})")
        # GUARD 2026-09-19: strict=False is required (frozen backbones are intentionally absent),
        # but it also silently tolerates a model built with the WRONG config -- the checkpoint's
        # tensors are then REJECTED as `unexpected` and the module keeps its pretrained init.
        # A model built without the code-vision prefix, for example, rejects the 6 merger tensors
        # as unexpected=6. Report both directions,
        # and name the modules involved so the mismatch is legible instead of a bare count.
        if res.unexpected_keys:
            _mods = sorted({k.split(".")[0] for k in res.unexpected_keys})
            logger.warning("load_checkpoint: %d checkpoint tensors were REJECTED (modules: %s). "
                           "The built model does not match the checkpoint -- check that the run's "
                           "use_code_expert / code_vision / action_dim reached the model config "
                           "(expected when loading a 3-expert checkpoint with use_code_expert=false).",
                           len(res.unexpected_keys), ", ".join(_mods))
        _miss = [k for k in res.missing_keys
                 if any(k.startswith(n + ".") for n, _ in self.extra_trainable_modules())]
        if _miss:
            _mods = sorted({k.split(".")[0] for k in _miss})
            logger.warning("load_checkpoint: %d TRAINABLE tensors absent from the checkpoint "
                           "(modules: %s) -- these keep their pretrained init. Expected only when "
                           "loading a checkpoint saved before that module was trained.",
                           len(_miss), ", ".join(_mods))
        return res

    @torch.no_grad()
    def _apply_skill_token_init(self):
        """Write compressed skill-token rows into embed_tokens / lm_head (SDAR is UNTIED, so both).
        Env-gated by AGENTICWAM_SKILL_TOKEN_INIT; a no-op when unset. The 54 ids live in the SPARE rows the
        SDAR matrix already has (tokenizer 151670 < vocab_size 151936), so NO resize is required."""
        import os as _os
        p = _os.environ.get("AGENTICWAM_SKILL_TOKEN_INIT")
        ce = getattr(self, "code_expert", None)
        if not p or ce is None or not _os.path.exists(p):
            return
        try:
            d = torch.load(p, map_location="cpu")
            ids = d["ids"].long()
            with torch.no_grad():
                emb = ce.embed_tokens.weight
                head = getattr(getattr(ce, "lm_head", None), "weight", None)
                if int(ids.max()) >= emb.shape[0]:
                    logger.error("skill-token init SKIPPED: id %d >= vocab rows %d",
                                 int(ids.max()), emb.shape[0])
                    return
                emb[ids] = d["embed"].to(device=emb.device, dtype=emb.dtype)
                if head is not None and int(ids.max()) < head.shape[0]:
                    head[ids] = d["lm_head"].to(device=head.device, dtype=head.dtype)
            logger.info("skill-token init APPLIED: %d rows -> ids %d..%d (embed%s, lm_head%s)",
                        len(ids), int(ids.min()), int(ids.max()),
                        tuple(emb.shape), tuple(head.shape) if head is not None else None)
        except Exception as e:
            logger.error("skill-token init FAILED: %s", e)

    def load_foundation(self, path: str):
        """Foundation-init (transfer learning): copy every SHAPE-MATCHING trainable tensor from a
        possibly DIFFERENT-action_dim checkpoint (backbone/video/code + the action transformer
        blocks) and SKIP shape-mismatched/absent ones (the action in/out projections + proprio
        encoder when action_dim differs). Unlike load_checkpoint (loud on a size mismatch), this is
        an EXPLICIT partial transfer for e.g. 64-D EEF -> 14-D delta-joint post-training."""
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        own = self.state_dict()
        keep, skip = {}, []
        for k, v in sd.items():
            if k in own and tuple(own[k].shape) == tuple(v.shape):
                keep[k] = v
            else:
                skip.append(k)
        res = self.load_state_dict(keep, strict=False)
        self._apply_skill_token_init()   # MUST run AFTER the foundation load, else the ckpt's
        # (untrained) rows overwrite the compressed skill embeddings -- the same ordering bug that
        # silently nullified E3's learned memory.
        logger.info(f"agenticwam load_foundation {path}: transferred {len(keep)}/{len(sd)} tensors; "
                    f"skipped {len(skip)} shape-mismatched/absent (e.g. {skip[:4]})")
        return res

    # ── weight loading ────────────────────────────────────────────────────────
    @torch.no_grad()
    def load_cosmos_video_checkpoint(self, state_dict: Dict[str, torch.Tensor], strict: bool = False):
        remap: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("transformer."):
                remap["dit.mot.experts.video.dit." + k[len("transformer."):]] = v
            elif k.startswith("vae."):
                remap["vae." + k[len("vae."):]] = v
            elif k.startswith("text_encoder."):
                remap["text_encoder." + k[len("text_encoder."):]] = v
            elif k.startswith("video_tokenizer."):
                remap["dit.video_tokenizer." + k[len("video_tokenizer."):]] = v
            # NOTE: `original_video_tokenizer.*` is the pristine 16-ch base tokenizer
            # (patch_in=25); the 48-ch fine-tuned one is `video_tokenizer.*` — we use that.
        # assign=True only when a meta-built (init_empty_weights) text encoder is present — it
        # materializes those params. Without it (training path), assign=False copies checkpoint
        # values into the existing fp32 params (upcasting bf16→fp32), keeping the DiT in fp32.
        assign = self.text_encoder is not None
        res = self.load_state_dict(remap, strict=strict, assign=assign)
        logger.info(f"cosmos video load: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
        return res

    # ── video embed / unembed (Cosmos path) ───────────────────────────────────
    @torch.no_grad()
    def _encode_video_latents(self, video: torch.Tensor) -> torch.Tensor:
        """Per-view VAE encode (matches the video pretraining). `video` is [B,3,T,H,W] in [-1,1]
        with T = n_cameras * T_per_view; each view is encoded independently (the VAE temporal
        conv must not cross view boundaries), then re-stacked along the temporal axis."""
        nc = self.n_cameras
        B, C, T, H, W = video.shape
        assert T % nc == 0, f"video T={T} not divisible by n_cameras={nc}"
        Tpv = T // nc
        vae_dtype = next(self.vae.parameters()).dtype
        per_view = (video.reshape(B, C, nc, Tpv, H, W)
                    .permute(0, 2, 1, 3, 4, 5).reshape(B * nc, C, Tpv, H, W))
        # Wan2.2 VAE (WanVideoVAE38): non-tiled per-view encode, as in the video pretraining.
        lat = self.vae.single_encode(per_view.to(vae_dtype))
        Cl, Tl, Hl, Wl = lat.shape[1:]
        lat = (lat.reshape(B, nc, Cl, Tl, Hl, Wl)
               .permute(0, 2, 1, 3, 4, 5).reshape(B, Cl, nc * Tl, Hl, Wl))
        return lat

    @torch.no_grad()
    def _decode_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Inverse of `_encode_video_latents`: per-view VAE decode of the generated future latents
        [B,48,nc*Tlpv,Hl,Wl] -> pixel video [B,3,nc*Tpix,Hpix,Wpix] in [-1,1] (views on the time axis)."""
        nc = self.n_cameras
        B, C, T, H, W = latents.shape
        assert T % nc == 0, f"latent frames {T} not divisible by n_cameras {nc}"
        Tpv = T // nc
        vae_dtype = next(self.vae.parameters()).dtype
        per_view = (latents.reshape(B, C, nc, Tpv, H, W)
                    .permute(0, 2, 1, 3, 4, 5).reshape(B * nc, C, Tpv, H, W))
        vid = self.vae.single_decode(per_view.to(vae_dtype))          # [B*nc,3,Tpix,Hpix,Wpix]
        C3, Tp, Hp, Wp = vid.shape[1:]
        vid = (vid.reshape(B, nc, C3, Tp, Hp, Wp)
               .permute(0, 2, 1, 3, 4, 5).reshape(B, C3, nc * Tp, Hp, Wp))
        return vid

    @staticmethod
    def _expand_token_sigmas(sigmas_per_frame: torch.Tensor, shape_info: dict) -> torch.Tensor:
        """(B,T) per-latent-frame sigma -> (B,L) per-token sigma. patch_t=1 so post_T==T; each
        latent frame owns post_H*post_W tokens."""
        spat = shape_info["post_H"] * shape_info["post_W"]
        return sigmas_per_frame.repeat_interleave(spat, dim=1)

    def _video_pre(self, noisy_video, video_sigmas_per_frame, prompt_embs, cond_mask):
        """Patchify + build the video expert aux (embedded_timestep, rope, text_ctx, idx)."""
        nc = self.n_cameras
        dit = self.video_expert.dit
        video_tokens, extra_pos_emb, shape_info = self.video_tokenizer.patchify(
            noisy_video, cond_mask, None, nc
        )
        B, L, _ = video_tokens.shape
        token_sigmas = self._expand_token_sigmas(video_sigmas_per_frame, shape_info)  # [B,L]
        rope = self.video_rope(noisy_video, n_cameras=nc)  # (cos, sin)
        temb, embedded_timestep = dit.time_embed(video_tokens, token_sigmas.reshape(-1))
        temb = temb.view(B, L, -1)
        embedded_timestep = embedded_timestep.view(B, L, -1)
        text_ctx = self._project_text(prompt_embs)
        if nc > 1 and not getattr(dit.config, "per_view_text", False):
            text_ctx = text_ctx.repeat(1, nc, 1)
        aux = {
            "embedded_timestep": embedded_timestep,
            "temb": temb,
            "rope": rope,
            "text_ctx": text_ctx,
            "n_cameras": nc,
            "video_idx": torch.arange(L, device=video_tokens.device),  # separate-expert seq is all video
            "non_video_idx": None,
            "extra_pos_emb": extra_pos_emb,
        }
        return video_tokens, aux, shape_info

    def _project_text(self, prompt_embs):
        dit = self.video_expert.dit
        if getattr(dit.config, "use_crossattn_projection", False):
            return dit.crossattn_proj(prompt_embs)
        return prompt_embs

    def _video_post(self, video_out, aux, shape_info):
        x = self.video_expert.dit.norm_out(video_out, aux["embedded_timestep"], aux["temb"])
        return self.video_tokenizer.unpatchify(x, shape_info)  # [B, C_out, T, H, W]

    # ── mask (AgenticWAM order: code -> video -> action) ─────────────────────────
    @torch.no_grad()
    def _history_ids(self, history_texts, device, max_len=192):
        """Tokenize completed-slice history text -> padded ids + valid mask (for the code memory)."""
        ce = self.code_expert; tok = ce.tokenizer
        pad = getattr(tok, "pad_token_id", 0) or 0
        seqs = []
        for h in history_texts:
            try: ids = list(tok.encode(h or "", add_special_tokens=False))
            except TypeError: ids = list(tok.encode(h or ""))
            # keep the TAIL under truncation: ids[:max_len] kept the OLDEST cycles and silently
            # dropped everything recent -- for an 18-cycle episode the slots never saw past ~cycle 3.
            seqs.append(ids[-max_len:] if ids else [pad])
        m = max(len(x) for x in seqs)
        ii = torch.tensor([x + [pad] * (m - len(x)) for x in seqs], dtype=torch.long, device=device)
        valid = torch.tensor([[True] * len(x) + [False] * (m - len(x)) for x in seqs], dtype=torch.bool, device=device)
        return ii, valid

    def _build_code_memory_state(self, history_texts, device, dtype, B):
        """E3: encode the episode's completed-slice history into the learned memory M[B,K,d].
        Self-gen (scheduled sampling): with prob p, encode the model's OWN predicted history."""
        ce = self.code_expert; mem = self.code_memory
        hist_ids, valid = self._history_ids(history_texts, device)
        p = float(getattr(self, "code_memory_selfgen_p", 0.0))
        mid = getattr(ce, "mask_token_id", None)
        if self.training and p > 0 and mid is not None and hist_ids.shape[1] > 0:
            sg = torch.rand(B, device=device) < p
            if bool(sg.any()):
                masked = torch.where(sg.unsqueeze(1) & valid, torch.full_like(hist_ids, mid), hist_ids)
                with torch.no_grad():
                    pred = ce._forward_code_only(masked).argmax(-1)
                hist_ids = torch.where(sg.unsqueeze(1) & valid, pred, hist_ids)
        hemb = ce.pre_dit(hist_ids)["tokens"].to(dtype)
        M = mem.init_state(B, device=device, dtype=dtype)
        return mem.write(M, hemb, valid_mask=valid)

    def _vit_prefix(self, video_pixels):
        """video_pixels: [B, nc*T, C, H, W] (or [B, nc, C, H, W] obs-only) in [-1,1].
        Returns ViT prefix tokens [B, P, 2048] or None when code_vision is disabled."""
        cv = getattr(self, "code_vision", None)
        if cv is None:
            return None
        nc = self.n_cameras
        # pipeline convention is channels-first video: [B, C=3, nc*T, H, W]
        # (training: T=9 pixel frames/view; inference obs_video: T=1). Slice the first
        # frame of each view block along the FRAME dim and emit [B, nc, C, H, W].
        assert video_pixels.dim() == 5 and video_pixels.shape[1] == 3, \
            f"expected [B,3,nc*T,H,W], got {tuple(video_pixels.shape)}"
        T = video_pixels.shape[2] // nc
        obs = torch.stack([video_pixels[:, :, v * T] for v in range(nc)], dim=1)
        # GRAD GATE FIX 2026-09-04: the trainer's dit-only policy keeps the TOP-LEVEL model in eval
        # mode (model.eval(); model.dit.train()), so `not self.training` was always True during
        # training and the merger -- registered as trainable via extra_trainable_modules() -- never
        # received a gradient (smoke: merger tensors byte-identical across saves). Gate on the
        # merger's OWN training flag (set by _apply_dit_only_train_mode, cleared by model.eval())
        # and respect the outer grad mode (eval decode under torch.no_grad() stays graph-free).
        # Tower params have requires_grad=False, so autograd records only the merger.
        _mg = getattr(getattr(cv, "vit", None), "merger", None)
        _train_merger = bool(_mg is not None and _mg.training and torch.is_grad_enabled())
        with (torch.enable_grad() if _train_merger else torch.no_grad()):
            return cv(obs)

    def _pre_dit_vis(self, input_ids, vit_tokens):
        """code_expert.pre_dit with an optional vision prefix, via the dummy-id trick
        (mirrors the code-memory prefix: RoPE covers P+L, then embeddings overwritten)."""
        if vit_tokens is None:
            return self.code_expert.pre_dit(input_ids), 0
        P = int(vit_tokens.shape[1])
        pad = getattr(self.code_expert.tokenizer, "pad_token_id", 0) or 0
        dummy = torch.full((input_ids.shape[0], P), pad, dtype=input_ids.dtype, device=input_ids.device)
        cp = self.code_expert.pre_dit(torch.cat([dummy, input_ids], dim=1))
        t = cp["tokens"].clone()
        if not os.environ.get("AGENTICWAM_ABLATE_VIT_PREFIX"):   # ablation: keep P pad-embedding rows (positions intact), drop the vision
            t[:, :P, :] = vit_tokens.to(t.dtype)
        cp["tokens"] = t
        return cp, P

    def _build_mot_attention_mask(self, video_seq_len, action_seq_len, device, code_seq_len=0,
                                  video_cond_tok=None, code_prefix_len=0):
        """WAM joint coupling (matches AgenticWAMJoint: "action attends to all video latent tokens").
        The action attends to the FULL video (obs cond frame + the predicted future frames), so it
        conditions on the world model's imagined rollout — not just the current observation. This is
        the coupling the 3-expert experiments proved necessary; inference co-denoises video+action.
        Video<->video full (Cosmos bidirectional diffusion); action<->action full."""
        c, v, a = int(code_seq_len), int(video_seq_len), int(action_seq_len)
        total = c + v + a
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        lengths = {"code": c, "video": v, "action": a}
        order = [m for m in self.mot.expert_order if lengths.get(m, 0) > 0]
        off, rng = 0, {}
        for m in order:
            rng[m] = (off, off + lengths[m]); off += lengths[m]
        vs, ve = rng["video"]; as_, ae = rng["action"]
        mask[vs:ve, vs:ve] = True                            # video <-> video (full)
        mask[as_:ae, as_:ae] = True                          # action <-> action
        mask[as_:ae, vs:ve] = True                           # action -> FULL video (obs + predicted future)
        if getattr(self, "_cfg_mode", "normal") == "drop_video_for_action":
            mask[as_:ae, vs:ve] = False                      # CFG dropout: action goes video-blind

        def _code_self(cs, cce):
            # code self-attention (identical rule set in both topologies)
            _co = getattr(getattr(self, "code_expert", None), "code_objective", "ar")
            if _co == "joint":
                mask[cs:cce, cs:cce] = True                    # LLaDA: fully-bidirectional
            elif _co == "blockdiff":
                _bl = int(getattr(self.code_expert, "code_block_length", 4))
                _nb = ((cce - cs) + _bl - 1) // _bl
                _bm = torch.tril(torch.ones((_nb, _nb), dtype=torch.bool, device=device))
                mask[cs:cce, cs:cce] = _bm.repeat_interleave(_bl, 0).repeat_interleave(_bl, 1)[: cce - cs, : cce - cs]
            else:
                mask[cs:cce, cs:cce] = torch.tril(
                    torch.ones((cce - cs, cce - cs), dtype=torch.bool, device=device))

        # AgenticWAM layout: the code stream = [vision prefix][instruction][canvas] is an attention island
        # (no cross-stream reads); the video keeps full self-attention and reads the code stream (the plan
        # steers the imagined future); the action reads everything.
        P = int(code_prefix_len)
        if c > 0:
            cs, cce = rng["code"]
            _code_self(cs + P, cce)                       # objective pattern over prompt+canvas
            if P > 0:
                mask[cs:cs + P, cs:cs + P] = True         # prefix: sealed conditioning source
                mask[cs + P:cce, cs:cs + P] = True        # prompt+canvas read the prefix
            mask[vs:ve, cs:cce] = True                    # video -> code stream (steering)
            mask[as_:ae, cs:cce] = True                   # action -> code
        return mask

    def training_loss(self, sample: Dict[str, Any]):
        # DeepSpeed casts the params to bf16 but disables torch.autocast (accelerator.autocast is a
        # no-op under DeepSpeed), so fp32 dataloader inputs would clash with bf16 weights. Run the
        # whole forward under an explicit autocast matching the param dtype (no-op when fp32).
        cdt = next(self.action_expert.parameters()).dtype
        with torch.autocast(device_type="cuda", dtype=cdt,
                            enabled=cdt in (torch.bfloat16, torch.float16)):
            return self._training_loss_impl(sample)

    def _training_loss_impl(self, sample: Dict[str, Any]):
        device = self.device
        # Compute dtype = the trainable params' dtype (bf16 under DeepSpeed, fp32 under autocast).
        # Dataloader tensors are fp32; cast action/proprio so they match the (bf16) expert weights.
        cdt = next(self.action_expert.parameters()).dtype
        video = sample["video"].to(device)                         # [B,3,T,H,W] multiview in [-1,1]
        action = sample["action"].to(device=device, dtype=cdt)     # [B,Ta,a_dim]
        proprio = sample.get("proprio", None)
        if proprio is not None:
            proprio = proprio.to(device=device, dtype=cdt)
        action_is_pad = sample.get("action_is_pad", None)
        num_cond_pixel = int(self.num_cond_frames)   # obs-conditioning (the cond-frame training)

        # 1. encode video + resolve text embeddings. Prefer precomputed Cosmos-Reason1 embeds
        #    (sample['prompt_embs'], cached offline like AgenticWAM's text cache) so the frozen 8.3B
        #    encoder stays out of the training model; else run it on-the-fly if present.
        latents = self._encode_video_latents(video)                # [B,48,T,H,W]
        prompt_embs = sample.get("prompt_embs")
        if prompt_embs is None:
            if self.text_encoder is None:
                raise ValueError("no text_encoder and sample has no precomputed 'prompt_embs'")
            with torch.no_grad():
                prompt_embs = self.text_encoder(sample["text"].to(device))
        prompt_embs = prompt_embs.to(device=device, dtype=latents.dtype)
        B, C, T, H, W = latents.shape
        nc = self.n_cameras
        assert T % nc == 0, f"latent frames {T} not divisible by n_cameras {nc}"
        T_lat_per_view = T // nc
        n_cond = self._pixel_to_latent(num_cond_pixel)             # cond latent frames per view

        # 2. video flow-matching: shifted sigma (Beta(7,1) for pretraining-fidelity, else uniform);
        #    x_sigma = (1-sigma)x0 + sigma*noise ; velocity target = noise - x0 ; cond frames clean.
        noise = torch.randn_like(latents)
        _rw_on = bool(getattr(self, "sigma_reweight", False))
        t, w_v, w_cond = self._sample_video_t_rw(B, device, latents.dtype)   # [B,1], [B], [B]
        shift = self.video_sigma_shift
        sigma = shift * t / (1 + (shift - 1) * t)                       # [B,1] shifted sigma
        sb = sigma.view(B, 1, 1, 1, 1)
        noisy = (1 - sb) * latents + sb * noise                        # [B,C,T,H,W]
        v_target = noise - latents                                     # velocity
        v_sig_pf = sigma.expand(B, T).clone()                          # per-frame sigma fed to the DiT
        cond_mask = torch.zeros(B, 1, T, H, W, device=device, dtype=latents.dtype)
        denoise = torch.ones(T, dtype=torch.bool, device=device)
        if n_cond > 0:
            for view in range(nc):
                i = view * T_lat_per_view
                noisy[:, :, i:i + n_cond] = latents[:, :, i:i + n_cond]
                v_sig_pf[:, i:i + n_cond] = self.cond_timestep
                cond_mask[:, :, i:i + n_cond] = 1.0
                denoise[i:i + n_cond] = False

        # 3. action: Wan flow-matching (matched pair with ActionDiT)
        noise_action = torch.randn_like(action)
        timestep_action = self.action_scheduler.sample_training_t(
            batch_size=B, device=device, dtype=action.dtype)
        noisy_action = self.action_scheduler.add_noise(action, noise_action, timestep_action)
        a_target = self.action_scheduler.training_target(action, noise_action, timestep_action)
        context, context_mask = self._action_context(prompt_embs, proprio)

        # 4. per-expert embed -> tokens + aux
        video_tokens, video_aux, shape_info = self._video_pre(noisy, v_sig_pf, prompt_embs, cond_mask)
        action_pre = self.action_expert.pre_dit(action_tokens=noisy_action, timestep=timestep_action,
                                                context=context, context_mask=context_mask)
        tokens = {"video": video_tokens, "action": action_pre["tokens"]}
        aux = {
            "video": video_aux,
            "action": {"t_mod": action_pre["t_mod"], "freqs": action_pre["freqs"],
                       "context": {"context": action_pre["context"], "mask": action_pre["context_mask"]}},
        }
        code_pre = code_labels = None
        code_seq_len = 0
        # CFG STREAM DROPOUT (env-gated: AGENTICWAM_CFG_DROPOUT="p_code,p_video"; absent -> path unchanged).
        # Per-rank-step mode draw; at eff.batch 96 the three regimes mix richly at the gradient level.
        #   drop_code  : code input becomes ALL-MASK (= the joint decode's start state, in-distribution
        #                null) -> trains p(action|video) and the code expert's unconditional branch.
        #   drop_video : the ACTION's attention to video is severed (mask block zeroed) -> the action
        #                head can only solve these samples by READING THE CODE. This branch is what
        #                forces the code channel to become load-bearing (armswap ±0.06% says the
        #                dependency does not form on its own).
        self._cfg_mode = "normal"
        _cfgd = os.environ.get("AGENTICWAM_CFG_DROPOUT")
        if _cfgd and torch.is_grad_enabled() and self.code_expert is not None and sample.get("policy_code") is not None:
            # os.urandom: immune to the trainer's deterministic reseeding (random.random() returned
            # the same value every step -> mode was stuck at "normal"; 0/14 firings at p=0.3).
            _pc, _pv = (float(x) for x in _cfgd.split(","))
            _r = int.from_bytes(os.urandom(2), "big") / 65536.0
            if _r < _pc:
                self._cfg_mode = "drop_code"
            elif _r < _pc + _pv:
                self._cfg_mode = "drop_video_for_action"
        if self.code_expert is not None and sample.get("policy_code") is not None:
            input_ids, code_labels = self.code_expert.build_code_batch(
                # Dual-instruction: prefer `prompt_code` (this episode's OWN randomised instruction,
                # emitted only when the dataset runs code_instruction_mode='episode'). Absent by
                # default -> falls back to `prompt`, i.e. behaviour is unchanged.
                instructions=(sample.get("prompt_nohist")
                              if (getattr(self, "code_memory_enabled", False) and sample.get("prompt_nohist") is not None)
                              else (sample.get("prompt_code") if sample.get("prompt_code") is not None
                                    else sample.get("prompt"))),
                codes=sample["policy_code"], device=device)
            _mem_M = None; _mem_K = 0
            if getattr(self, "code_memory_enabled", False) and self.code_memory is not None:
                _bs = input_ids.shape[0]; _mem_K = int(self.code_memory.K)
                _pad = getattr(self.code_expert.tokenizer, "pad_token_id", 0) or 0
                _dummy = torch.full((_bs, _mem_K), _pad, dtype=input_ids.dtype, device=device)
                input_ids = torch.cat([_dummy, input_ids], dim=1)          # K read-only prefix slots
                code_labels = torch.cat([torch.full((_bs, _mem_K), -100, dtype=code_labels.dtype, device=device), code_labels], dim=1)
                _mem_M = self._build_code_memory_state(sample.get("code_history") or [""] * _bs, device, latents.dtype, _bs)
            if getattr(self, "_cfg_mode", "normal") == "drop_code":
                _mid = getattr(self.code_expert, "mask_token_id", None)
                if _mid is not None:
                    input_ids = torch.full_like(input_ids, _mid)
            _vit = self._vit_prefix(sample["video"].to(device)) if getattr(self, "code_vision", None) is not None else None
            if _vit is not None:
                code_labels = torch.cat([torch.full((code_labels.shape[0], _vit.shape[1]), -100,
                                                    dtype=code_labels.dtype, device=code_labels.device), code_labels], dim=1)
                code_pre, _vitP = self._pre_dit_vis(input_ids, _vit)
            else:
                _vitP = 0
                code_pre = self.code_expert.pre_dit(input_ids)               # RoPE covers K+L cleanly
            if _mem_K > 0:
                # FIX 2026-09-03: the K memory slots sit AFTER the ViT prefix ([ViT P][mem K][prompt][canvas]);
                # writing at [:K] overwrote the first K ViT tokens and left the slots as pad embeddings.
                _t = code_pre["tokens"].clone()
                _t[:, _vitP:_vitP + _mem_K, :] = self.code_memory.read(_mem_M).to(_t.dtype)   # inject memory into ITS slots
                code_pre["tokens"] = _t
            code_seq_len = int(code_pre["tokens"].shape[1])
            tokens["code"] = code_pre["tokens"]
            aux["code"] = {"t_mod": code_pre["t_mod"], "freqs": code_pre["freqs"], "context": None}

        # 5. joint mask + MoT (action attends to the FULL video: obs + predicted future)
        _vct = None
        _vp = locals().get("_vitP", 0) or 0
        _mk = locals().get("_mem_K", 0) or 0
        mask = self._build_mot_attention_mask(video_tokens.shape[1], action_pre["tokens"].shape[1],
                                              device, code_seq_len, video_cond_tok=_vct,
                                              code_prefix_len=(_vp + _mk) if _vp else 0)   # memory slots join the sealed prefix
        out = self.mot(tokens, aux, mask)

        # 6. video velocity loss (denoise frames only; uniform weight — Beta sampling already
        #    shapes the noise-level distribution).
        pred_video = self._video_post(out["video"], video_aux, shape_info)
        pv, tv = pred_video[:, :, denoise], v_target[:, :, denoise]
        if _rw_on:   # per-sample MSE so the video objective can be importance-weighted
            _v_ps = ((pv.float() - tv.float()) ** 2).flatten(1).mean(1)          # [B]
            loss_video = (_v_ps * w_v.to(_v_ps.dtype)).mean()
        else:
            loss_video = F.mse_loss(pv.float(), tv.float())

        # 7. action loss
        pred_action = self.action_expert.post_dit(out["action"], action_pre)
        a_tok = F.mse_loss(pred_action.float(), a_target.float(), reduction="none").mean(dim=2)  # [B,Ta]
        if action_is_pad is not None:
            valid = (~action_is_pad.to(device)).to(a_tok.dtype)
            a_ps = (a_tok * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        else:
            a_ps = a_tok.mean(1)
        aw = self.action_scheduler.training_weight(timestep_action).to(a_ps.device, a_ps.dtype).view(-1)
        loss_action = ((a_ps * aw * w_cond.to(a_ps.dtype)).mean() if _rw_on
                       else (a_ps * aw).mean())

        loss = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {"loss_video": float(loss_video.detach()), "loss_action": float(loss_action.detach())}
        _cm = getattr(self, "_cfg_mode", "normal")
        if os.environ.get("AGENTICWAM_CFG_DROPOUT"):
            loss_dict["cfg_drop_code"] = 1.0 if _cm == "drop_code" else 0.0
            loss_dict["cfg_drop_video"] = 1.0 if _cm == "drop_video_for_action" else 0.0
        if _rw_on:   # sigma/weight instrumentation for the reweighting experiment
            loss_dict["sigma_mean"] = float(sigma.mean().detach())
            loss_dict["sigma_lo_frac"] = float((sigma < 0.3).float().mean().detach())
            loss_dict["w_v_mean"] = float(w_v.mean().detach())
            loss_dict["w_cond_mean"] = float(w_cond.mean().detach())

        # 8. code CE
        if code_pre is not None and code_labels is not None and "code" in out:
            code_logits = self.code_expert.post_dit(out["code"])
            code_out = self.code_expert.code_loss(code_logits, code_labels)
            _code_term = code_out["loss"]
            # RL HOOK (wd1-style reward-weighted MLE): when `_rl_weights` [B] is set, the code term
            # becomes sum_g w_g * per-sample-CE_g instead of the mean. Same single cross_entropy op,
            # so the autograd graph stays identical across ranks (DDP/ZeRO requirement).
            # `_rl_weights` is None everywhere except the RL trainer -> default path byte-identical.
            _rlw = getattr(self, "_rl_weights", None)
            if _rlw is not None:
                _lg = code_logits.float()
                _ce = F.cross_entropy(_lg.reshape(-1, _lg.size(-1)), code_labels.reshape(-1),
                                      ignore_index=-100, reduction="none").view(code_labels.shape)
                _vm = (code_labels != -100).to(_ce.dtype)
                _c_ps = (_ce * _vm).sum(1) / _vm.sum(1).clamp(min=1.0)
                _code_term = (_c_ps * _rlw.to(_c_ps.dtype)).sum()
                # entropy bonus (RL-only, default 0): counteracts weighted-MLE's entropy contraction.
                _ec = float(getattr(self, "_rl_ent_coef", 0.0) or 0.0)
                if _ec > 0:
                    _lp = _lg.log_softmax(-1)
                    _H = -(( _lp.exp() * _lp).sum(-1) * _vm).sum() / _vm.sum().clamp(min=1.0)
                    _code_term = _code_term - _ec * _H
            elif _rw_on and getattr(self.code_expert, "code_objective", "ar") in ("joint", "blockdiff"):
                # per-sample CE (same single cross_entropy op -> autograd structure stays
                # identical across ranks, which DDP/ZeRO requires); labels -100 = ignore.
                _lg = code_logits.float()
                _ce = F.cross_entropy(_lg.reshape(-1, _lg.size(-1)), code_labels.reshape(-1),
                                      ignore_index=-100, reduction="none").view(code_labels.shape)
                _vm = (code_labels != -100).to(_ce.dtype)
                _c_ps = (_ce * _vm).sum(1) / _vm.sum(1).clamp(min=1.0)                # [B]
                _code_term = (_c_ps * w_cond.to(_c_ps.dtype)).mean()
            loss = loss + self.loss_lambda_code * _code_term
            loss_dict["loss_code"] = float(code_out["loss"].detach())
            loss_dict["code_acc"] = float(code_out["acc"].detach())
            # EOS-excluded accuracy, emitted by code_loss whenever the EOS labels are non-trivial
            # (pad-EOS window/weight, or a fixed code_canvas). Under canvas 150 about 36% of canvas
            # labels are EOS and trivially predictable, so code_acc is inflated and NOT comparable to
            # the batch-max-aligned runs; code_acc_noneos is the comparable curve.
            if "acc_noneos" in code_out:
                loss_dict["code_acc_noneos"] = float(code_out["acc_noneos"].detach())

        return loss, loss_dict

    def _action_context(self, prompt_embs, proprio):
        """Cross-attn context for the action expert = projected Reason1 text (1024) [+ proprio(1024)].
        Returns (context [B,L,1024], context_mask [B,L]). ActionDiT.pre_dit applies its own
        text_embedding on top."""
        ctx = self._project_text(prompt_embs)
        mask = torch.ones(ctx.shape[:2], dtype=torch.bool, device=ctx.device)
        if proprio is not None and self.proprio_encoder is not None:
            p = self.proprio_encoder(proprio[:, 0, :].to(ctx.dtype))  # [B,1024]
            ctx = torch.cat([ctx, p.unsqueeze(1)], dim=1)
            mask = torch.cat([mask, torch.ones(mask.shape[0], 1, dtype=torch.bool, device=ctx.device)], dim=1)
        return ctx, mask
    def _sample_video_t_rw(self, B: int, device, dtype):
        """Video flow-matching t plus per-objective IMPORTANCE WEIGHTS (default: disabled).

        Motivation: the video sigma is shared by the video objective AND by whatever the code /
        action experts read through mixed attention. With the shipped Beta(7,1)+shift=5 setting
        ~95% of training samples land at sigma>0.9, so the policy almost never sees an informative
        future while the inference loop sweeps sigma 1->0. Sampling from a defensive mixture and
        importance-weighting each objective back to its own target decouples the two WITHOUT any
        extra forward, extra tokens, or inference change:

            t ~ q = alpha*P_cur + (1-alpha)*P_inf
            w_v    = P_cur / q   applied to L_video   -> unbiased for the ORIGINAL video objective
            w_cond = P_inf / q   applied to L_action/L_code -> trains under the INFERENCE density

        P_cur = the configured Beta(a,b)*noise_s density; P_inf = U(0,1), which is what the
        inference sweep traverses (see _video_inference_sigmas: t is linspace(1,0) pre-shift).
        Because q contains both targets, weights are bounded by 1/alpha and 1/(1-alpha).

        Returns (t [B,1], w_v [B], w_cond [B]). Disabled -> original t, weights = 1.
        """
        if not bool(getattr(self, "sigma_reweight", False)) or self.video_noise_beta is None:
            one = torch.ones(B, device=device, dtype=dtype)
            return self._sample_video_t(B, device, dtype), one, one

        a, b, sc = self.video_noise_beta
        alpha = float(getattr(self, "sigma_reweight_alpha", 0.5))
        beta = torch.distributions.Beta(torch.tensor(float(a)), torch.tensor(float(b)))
        n_cur = int(round(alpha * B))
        if not bool(getattr(self, "sigma_reweight_stratified", True)):
            n_cur = int(torch.bernoulli(torch.full((B,), alpha)).sum().item())
        n_cur = max(0, min(B, n_cur))
        parts = []
        if n_cur > 0:
            parts.append(beta.sample((n_cur, 1)).to(device=device, dtype=torch.float32) * float(sc))
        if B - n_cur > 0:
            parts.append(torch.rand(B - n_cur, 1, device=device, dtype=torch.float32))
        t = torch.cat(parts, 0)[torch.randperm(B, device=device)].clamp(1e-4, 1 - 1e-4)

        # densities of the two targets at the drawn t (float32 for stability)
        y = (t / float(sc)).clamp(1e-6, 1 - 1e-6)
        p_cur = torch.exp(beta.log_prob(y.cpu()).to(device)) / float(sc)        # p of sc*Beta(a,b)
        p_cur = torch.where(t <= float(sc), p_cur, torch.zeros_like(p_cur)).squeeze(-1)
        p_inf = torch.ones_like(p_cur)                                          # U(0,1)
        q = (alpha * p_cur + (1.0 - alpha) * p_inf).clamp(min=1e-8)
        clip = float(getattr(self, "sigma_reweight_clip", 4.0))
        w_v = (p_cur / q).clamp(max=clip)
        w_cond = (p_inf / q).clamp(max=clip)
        if bool(getattr(self, "sigma_reweight_normalize", True)):
            w_v = w_v / w_v.mean().clamp(min=1e-6)
            w_cond = w_cond / w_cond.mean().clamp(min=1e-6)
        return (t.to(dtype), w_v.detach().to(dtype), w_cond.detach().to(dtype))


    def _sample_video_t(self, B: int, device, dtype) -> torch.Tensor:
        """Per-sample flow-matching t in (0,1). Beta(alpha,beta)*noise_s when configured
        (the pretraining uses (7.0,1.0,0.999)); else uniform."""
        if self.video_noise_beta is not None:
            a, b, s = self.video_noise_beta
            dist = torch.distributions.Beta(torch.tensor(float(a)), torch.tensor(float(b)))
            t = dist.sample((B, 1)).to(device=device, dtype=dtype) * float(s)
        else:
            t = torch.rand(B, 1, device=device, dtype=dtype)
        return t.clamp(1e-4, 1 - 1e-4)

    @staticmethod
    def _pixel_to_latent(num_pixel: int) -> int:
        return 0 if num_pixel <= 0 else (num_pixel - 1) // 4 + 1  # VAE temporal 4x

    def _video_inference_sigmas(self, num_steps, device, dtype, shift=None):
        """Rectified-flow shifted-sigma sweep from ~1 (noise) to 0 (clean), matching training's
        sigma = shift*t/(1+(shift-1)*t). Returns [num_steps+1] decreasing sigmas."""
        shift = float(self.video_sigma_shift if shift is None else shift)
        t = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
        return shift * t / (1.0 + (shift - 1.0) * t)

    # ── joint video+action rollout inference (WAM: predict the full future video) ─────────
    @torch.no_grad()
    def infer_action(self, obs_video, proprio, prompt_embs, action_horizon,
                     num_video_frames: int = 9, num_inference_steps: int = 20, sigma_shift=None,
                     return_video: bool = False):
        """Predict a 64-D action chunk by JOINTLY denoising the full future video + the action
        (World-Action-Model rollout; matches AgenticWAMJoint.infer_action and the training coupling
        where the action attends to the FULL video). Each view's obs frame(s) are the clean
        conditioning (sigma=cond_timestep); the remaining latent frames/view start as noise and are
        generated, and the action co-denoises while attending to that imagined future.

        obs_video:        [1,3,nc,H,W] multiview obs in [-1,1] (1 obs frame/view, views on time).
        num_video_frames: pixel frames/view to predict (obs + future); 9 (training horizon) ->
                          3 latent frames/view (1 obs cond + 2 generated).
        proprio:          [1,proprio_dim] or [1,1,proprio_dim] current state.
        prompt_embs:      [1,N,100352] Cosmos-Reason1 embed for the task instruction.
        returns:          [action_horizon, action_dim] fp32 CPU tensor.
        """
        self.dit.eval()
        device = self.device
        cdt = next(self.action_expert.parameters()).dtype
        adim = self.action_expert.action_dim
        nc = self.n_cameras
        if proprio is not None and proprio.dim() == 2:
            proprio = proprio.unsqueeze(1)
        with torch.autocast(device_type="cuda", dtype=cdt,
                            enabled=cdt in (torch.bfloat16, torch.float16)):
            # 1. encode the obs frame(s) -> per-view clean cond latents
            obs_latents = self._encode_video_latents(obs_video.to(device))
            _vit_pref = self._vit_prefix(obs_video.to(device))   # [1,Cl,nc*n_cond_lat,Hl,Wl]
            Cl, Hl, Wl = obs_latents.shape[1], obs_latents.shape[3], obs_latents.shape[4]
            n_cond_lat = obs_latents.shape[2] // nc
            Tlpv = max(self._pixel_to_latent(int(num_video_frames)), n_cond_lat)   # latent frames/view
            prompt_embs = prompt_embs.to(device=device, dtype=cdt)
            if proprio is not None:
                proprio = proprio.to(device=device, dtype=cdt)

            # 2. init the full video latents (noise); clamp each view's obs frame(s) to the clean latent
            latents_video = torch.randn(1, Cl, nc * Tlpv, Hl, Wl, device=device, dtype=cdt)
            denoise = torch.ones(nc * Tlpv, dtype=torch.bool, device=device)
            for view in range(nc):
                d0, s0 = view * Tlpv, view * n_cond_lat
                latents_video[:, :, d0:d0 + n_cond_lat] = obs_latents[:, :, s0:s0 + n_cond_lat]
                denoise[d0:d0 + n_cond_lat] = False
            dmask = denoise.view(1, 1, nc * Tlpv, 1, 1)
            latents_action = torch.randn(1, action_horizon, adim, device=device, dtype=cdt)

            # 3. schedules: video = the pretraining's UniPC sampler (self.video_scheduler:
            #    set_timesteps + step, solver_order=2, predict_x0, velocity=noise-x0 — matches the
            #    training convention). A naive Euler sweep undersamples the low-sigma detail region
            #    (training σ~Beta(7,1) concentrates high) → noise; UniPC fixes it. Action = its own
            #    continuous-flow scheduler. Both advance one step per iteration (joint denoise).
            context, context_mask = self._action_context(prompt_embs, proprio)
            # restore the pristine training schedule (set_timesteps mutates self.sigmas in place;
            # without this the schedule compounds across calls -> degenerate -> black video)
            self.video_scheduler.sigmas = self._vs_sigmas0.clone().to(device)
            self.video_scheduler.timesteps = self._vs_timesteps0.clone().to(device)
            self.video_scheduler.set_timesteps(num_inference_steps, device=device)
            ts_a, deltas_a = self.action_scheduler.build_inference_schedule(
                num_inference_steps, device, cdt, shift_override=sigma_shift)

            # 4. JOINT denoise: video (UniPC) + action co-evolve; the action attends to the FULL video
            for step_index, (t_a, d_a) in enumerate(zip(ts_a, deltas_a)):
                sigma_v = self.video_scheduler.sigmas[step_index].to(device=device, dtype=cdt)
                ts_v = self.video_scheduler.timesteps[step_index]
                v_sig_pf = torch.full((1, nc * Tlpv), float(sigma_v), device=device, dtype=cdt)
                cond_mask = torch.zeros(1, 1, nc * Tlpv, Hl, Wl, device=device, dtype=cdt)
                for view in range(nc):
                    d0 = view * Tlpv
                    v_sig_pf[:, d0:d0 + n_cond_lat] = self.cond_timestep
                    cond_mask[:, :, d0:d0 + n_cond_lat] = 1.0
                video_tokens, video_aux, shape_info = self._video_pre(latents_video, v_sig_pf, prompt_embs, cond_mask)
                ts_a_ = t_a.reshape(1).to(device=device, dtype=cdt)
                action_pre = self.action_expert.pre_dit(action_tokens=latents_action, timestep=ts_a_,
                                                        context=context, context_mask=context_mask)
                tokens = {"video": video_tokens, "action": action_pre["tokens"]}
                aux = {"video": video_aux,
                       "action": {"t_mod": action_pre["t_mod"], "freqs": action_pre["freqs"],
                                  "context": {"context": action_pre["context"], "mask": action_pre["context_mask"]}}}
                _vct = None
                mask = self._build_mot_attention_mask(video_tokens.shape[1], action_pre["tokens"].shape[1],
                                                      device, code_seq_len=0, video_cond_tok=_vct)
                out = self.mot(tokens, aux, mask)
                # video: UniPC step on the predicted velocity (scheduler converts to x0 via predict_x0);
                # only generated frames evolve — obs cond frames are re-clamped to the clean latent.
                pred_video = self._video_post(out["video"], video_aux, shape_info)
                if not os.environ.get("AGENTICWAM_ABLATE_FREEZE_VIDEO"):
                    res = self.video_scheduler.step(pred_video, ts_v, latents_video, step_index)
                    latents_video = res[0] if isinstance(res, (tuple, list)) else res
                for view in range(nc):
                    d0, s0 = view * Tlpv, view * n_cond_lat
                    latents_video[:, :, d0:d0 + n_cond_lat] = obs_latents[:, :, s0:s0 + n_cond_lat]
                # action: its own scheduler step
                pred_action = self.action_expert.post_dit(out["action"], action_pre)
                latents_action = self.action_scheduler.step(pred_action, d_a, latents_action)
            action_out = latents_action[0].detach().float().cpu()
            if return_video:
                # decode the model's IMAGINED future video (the WAM rollout) for visualization
                gen_video = self._decode_video_latents(latents_video).detach().float().cpu()
                return action_out, gen_video          # ([Ta,adim], [1,3,nc*Tpix,Hpix,Wpix] in [-1,1])
        return action_out

    @torch.no_grad()
    @torch.no_grad()
    def _generate_code_obs_blockdiff(self, instruction, obs_video_tokens, obs_video_aux,
                                     action_dummy_pre, max_new_tokens: int = 128,
                                     denoise_steps: int = 0, conf_threshold: float = 0.9):
        """Global low-confidence-dynamic absorbing decode (matches the global random-mask training):
        the code is a fixed-length <|MASK|> slot; each pass predicts ALL masked positions and unmasks
        those with confidence > conf_threshold (or, if none clear it, only the single most-confident
        token, to guarantee progress), RE-predicting the rest each pass so newly-committed context
        corrects the remaining tokens. Block-causal attention as trained; one MoT forward per pass,
        no growing sequence / no KV cache. Avoids the block-by-block commit cascade that garbled output."""
        ce = self.code_expert; device = self.device
        mid = ce.mask_token_id
        prompt = ("" if os.environ.get("AGENTICWAM_CODE_NO_INSTRUCTION", "0") == "1" else f"# Instruction: {instruction}\n")
        try:
            pids = ce.tokenizer.encode(prompt, add_special_tokens=False)
        except TypeError:
            pids = ce.tokenizer.encode(prompt)
        eos = getattr(ce.tokenizer, "eos_token_id", None)
        plen = len(pids)
        gen_len = int(max_new_tokens)
        cap = int(ce.freqs.shape[0])
        if plen + gen_len > cap:
            gen_len = cap - plen
        ids_t = torch.tensor([pids + [mid] * gen_len], dtype=torch.long, device=device)
        v_len = int(obs_video_tokens.shape[1]); a_len = int(action_dummy_pre["tokens"].shape[1])
        a_aux = {"t_mod": action_dummy_pre["t_mod"], "freqs": action_dummy_pre["freqs"],
                 "context": {"context": action_dummy_pre["context"], "mask": action_dummy_pre["context_mask"]}}
        c_len = plen + gen_len
        _vct_all = None
        c_len = c_len + (int(_vit_pref.shape[1]) if _vit_pref is not None else 0)
        mask = self._build_mot_attention_mask(v_len, a_len, device, code_seq_len=c_len,
                                              video_cond_tok=_vct_all,
                                              code_prefix_len=(int(_vit_pref.shape[1]) if _vit_pref is not None else 0))

        def _code_logits():
            code_pre, _vitP = self._pre_dit_vis(ids_t, _vit_pref)
            tokens = {"video": obs_video_tokens, "code": code_pre["tokens"], "action": action_dummy_pre["tokens"]}
            aux = {"video": obs_video_aux,
                   "code": {"t_mod": code_pre["t_mod"], "freqs": code_pre["freqs"], "context": None},
                   "action": a_aux}
            return ce.post_dit(self.mot(tokens, aux, mask)["code"])[0][_vitP:]   # [plen+gen_len, vocab]; ViT prefix rows sliced off

        # Decode-ablation knobs (env-gated; defaults == original greedy 1-token/pass behavior):
        #   AGENTICWAM_DEC_TEMP  (0=argmax)  AGENTICWAM_DEC_TOPP (1.0=off)  AGENTICWAM_DEC_REP (1.0=off, HF-style
        #   repetition penalty over already-committed gen tokens)  AGENTICWAM_DEC_ADJ (1=ban committing a
        #   token equal to an already-committed immediate neighbor)  AGENTICWAM_DEC_KPP (tokens/pass).
        _temp = float(os.environ.get("AGENTICWAM_DEC_TEMP", "0") or 0)
        _topp = float(os.environ.get("AGENTICWAM_DEC_TOPP", "1.0") or 1.0)
        _rep = float(os.environ.get("AGENTICWAM_DEC_REP", "1.0") or 1.0)
        _adj = os.environ.get("AGENTICWAM_DEC_ADJ", "0") == "1"
        _kpp = max(1, int(os.environ.get("AGENTICWAM_DEC_KPP", "1") or 1))
        max_passes = int(denoise_steps) if denoise_steps else gen_len
        for _ in range(max_passes):
            masked = (ids_t[0, plen:] == mid).nonzero(as_tuple=True)[0] + plen
            if masked.numel() == 0:
                break
            logits = _code_logits()
            lg = logits[masked].float()
            if _rep != 1.0:
                committed = ids_t[0, plen:][ids_t[0, plen:] != mid]
                if committed.numel() > 0:
                    uniq = torch.unique(committed)
                    sel = lg[:, uniq]
                    lg[:, uniq] = torch.where(sel > 0, sel / _rep, sel * _rep)
            if _adj:
                lo, hi = plen, plen + gen_len
                for r in range(masked.numel()):
                    p = int(masked[r])
                    for nb in (p - 1, p + 1):
                        if lo <= nb < hi:
                            t = int(ids_t[0, nb])
                            if t != mid:
                                lg[r, t] = float("-inf")
            probs = lg.softmax(-1)
            if _temp and _temp > 0:
                pt = (lg / _temp).softmax(-1)
                if _topp < 1.0:
                    sp, si = pt.sort(-1, descending=True)
                    cum = sp.cumsum(-1)
                    keep = (cum - sp) < _topp
                    sp = sp * keep
                    sp = sp / sp.sum(-1, keepdim=True).clamp_min(1e-9)
                    ch = torch.multinomial(sp, 1)
                    tok = si.gather(-1, ch).squeeze(-1)
                else:
                    tok = torch.multinomial(pt, 1).squeeze(-1)
                conf = probs.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
            else:
                conf, tok = probs.max(-1)
            k = min(_kpp, int(masked.numel()))
            for j in torch.topk(conf, k).indices.tolist():
                ids_t[0, int(masked[j])] = int(tok[j])
        rem = (ids_t[0, plen:] == mid).nonzero(as_tuple=True)[0] + plen
        if rem.numel() > 0:
            logits = _code_logits()
            ids_t[0, rem] = logits[rem].argmax(-1).to(ids_t.dtype)
        gen = [t for t in ids_t[0, plen:].tolist() if t != mid]
        if eos is not None and eos in gen:
            if os.environ.get("AGENTICWAM_EOS_RUN_TRUNC", "0") == "1":
                # terminal-run truncation: strip the trailing EOS run, drop isolated interior
                # EOS -- a stray early EOS commit then costs one token, not the whole decoded
                # suffix (first-EOS truncation chops everything after it)
                while gen and gen[-1] == eos:
                    gen.pop()
                gen = [t for t in gen if t != eos]
            else:
                gen = gen[:gen.index(eos)]
        try:
            return ce.tokenizer.decode(gen, skip_special_tokens=True).strip()
        except Exception:
            return ""

    def _generate_code_obs(self, instruction, obs_video_tokens, obs_video_aux,
                           action_dummy_pre, max_new_tokens: int = 128):
        """AR-generate policy code conditioned on the (clean) obs video via the joint MoT
        (code->video attention, matching the code-in-middle training mask). `action_dummy_pre`
        is a 1-token placeholder so the MoT (which requires ALL experts present) can run; the
        code does not attend to it (mask). Greedy decode until EOS or the token cap."""
        ce = self.code_expert
        if getattr(ce, "code_objective", "ar") == "blockdiff":
            return self._generate_code_obs_blockdiff(
                instruction, obs_video_tokens, obs_video_aux, action_dummy_pre, max_new_tokens)
        device = self.device
        prompt = ("" if os.environ.get("AGENTICWAM_CODE_NO_INSTRUCTION", "0") == "1" else f"# Instruction: {instruction}\n")
        try:
            ids = ce.tokenizer.encode(prompt, add_special_tokens=False)
        except TypeError:
            ids = ce.tokenizer.encode(prompt)
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        prompt_len = ids_t.shape[1]
        eos = getattr(ce.tokenizer, "eos_token_id", None)
        v_len = int(obs_video_tokens.shape[1])
        a_len = int(action_dummy_pre["tokens"].shape[1])
        a_aux = {"t_mod": action_dummy_pre["t_mod"], "freqs": action_dummy_pre["freqs"],
                 "context": {"context": action_dummy_pre["context"], "mask": action_dummy_pre["context_mask"]}}
        for _ in range(int(max_new_tokens)):
            if ids_t.shape[1] > ce.freqs.shape[0]:
                break
            code_pre, _vitP = self._pre_dit_vis(ids_t, _vit_pref)
            c_len = int(code_pre["tokens"].shape[1])
            _vct_all = None
            mask = self._build_mot_attention_mask(v_len, a_len, device, code_seq_len=c_len,
                                                  video_cond_tok=_vct_all,
                                                  code_prefix_len=locals().get("_vitP", 0) or 0)
            tokens = {"video": obs_video_tokens, "code": code_pre["tokens"], "action": action_dummy_pre["tokens"]}
            aux = {"video": obs_video_aux,
                   "code": {"t_mod": code_pre["t_mod"], "freqs": code_pre["freqs"], "context": None},
                   "action": a_aux}
            out = self.mot(tokens, aux, mask)
            logits = ce.post_dit(out["code"])
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids_t = torch.cat([ids_t, nxt], dim=1)
            if eos is not None and bool((nxt == eos).all()):
                break
        gen = ids_t[0, prompt_len:].tolist()
        try:
            return ce.tokenizer.decode(gen, skip_special_tokens=True).strip()
        except Exception:
            return bytes([t for t in gen if 0 <= t < 256]).decode("utf-8", errors="replace")

    @torch.no_grad()
    def _infer_action_joint(self, obs_video, proprio, prompt_embs, action_horizon,
                            num_video_frames: int = 9, num_inference_steps: int = 20,
                            sigma_shift=None, instruction=None, max_code_tokens: int = 128,
                            return_video: bool = False):
        """FULLY JOINT tri-modal denoising (LLaDA code). Initialise video~noise, action~noise, and the
        code as an all-<MASK> slot; over the SHARED T steps, ONE MoT forward co-updates all three:
        video/action by flow-matching, code by confidence-unmasking gen_len/T tokens per step. The code
        co-refines with the (denoising) future video (E6) instead of being generated first and frozen."""
        self.dit.eval(); device = self.device
        cdt = next(self.action_expert.parameters()).dtype
        adim = self.action_expert.action_dim; nc = self.n_cameras
        ce = self.code_expert; mid = ce.mask_token_id
        if proprio is not None and proprio.dim() == 2:
            proprio = proprio.unsqueeze(1)
        with torch.autocast(device_type="cuda", dtype=cdt,
                            enabled=cdt in (torch.bfloat16, torch.float16)):
            obs_latents = self._encode_video_latents(obs_video.to(device))
            _vit_pref = self._vit_prefix(obs_video.to(device))
            Cl, Hl, Wl = obs_latents.shape[1], obs_latents.shape[3], obs_latents.shape[4]
            n_cond_lat = obs_latents.shape[2] // nc
            Tlpv = max(self._pixel_to_latent(int(num_video_frames)), n_cond_lat)
            prompt_embs = prompt_embs.to(device=device, dtype=cdt)
            if proprio is not None:
                proprio = proprio.to(device=device, dtype=cdt)
            context, context_mask = self._action_context(prompt_embs, proprio)
            if instruction is None:
                instruction = ""
            prompt = ("" if os.environ.get("AGENTICWAM_CODE_NO_INSTRUCTION", "0") == "1" else f"# Instruction: {instruction}\n")
            # [dataprep] train/eval prompt parity: datasets built with a helper-library preamble
            # (robotwin/dataprep) train on "# Instruction / # Helper library / # Next",
            # so generation must see the same prompt. AGENTICWAM_CODE_PREAMBLE_FILE (the task's preamble.txt)
            # and AGENTICWAM_CODE_NO_HISTORY are exported by robotwin/hybrid_eval/launch.py.
            _pf = os.environ.get("AGENTICWAM_CODE_PREAMBLE_FILE")
            if _pf and os.path.exists(_pf):
                _hist = ""   # completed slices are not tracked at generation time
                if getattr(self, "code_memory_enabled", False) or os.environ.get("AGENTICWAM_CODE_NO_HISTORY", "0") == "1":
                    # models trained on prompt_nohist / code_slices_no_history: no "# Completed:" block
                    # (inserting one collapses their accuracy).
                    prompt = (f"# Instruction: {instruction}\n# Helper library:\n"
                              f"{open(_pf).read()}# Next:\n")
                else:
                    prompt = (f"# Instruction: {instruction}\n# Helper library:\n"
                              f"{open(_pf).read()}# Completed:\n{_hist}# Next:\n")
            try:
                pids = ce.tokenizer.encode(prompt, add_special_tokens=False)
            except TypeError:
                pids = ce.tokenizer.encode(prompt)
            plen = len(pids)
            gen_len = int(max_code_tokens)
            cap = int(ce.freqs.shape[0])
            if plen + gen_len > cap:
                gen_len = cap - plen
            # E3: prepend K read-only memory slots (mirrors training's dummy-prefix trick).
            _mem_K = 0; _mem_M = None
            if getattr(self, "code_memory_enabled", False) and getattr(self, "code_memory", None) is not None:
                _mem_K = int(self.code_memory.K)
                _mpad = getattr(ce.tokenizer, "pad_token_id", 0) or 0
                _hist = getattr(self, "_mem_code_history", "") or ""
                _mem_M = self._build_code_memory_state([_hist], device, cdt, 1)
                pids = [_mpad] * _mem_K + pids
                plen = len(pids)
            code_ids = torch.tensor([pids + [mid] * gen_len], dtype=torch.long, device=device)
            _oc = os.environ.get("AGENTICWAM_ORACLE_CODE_FILE")
            if _oc and os.path.exists(_oc):
                _ot = ce.tokenizer.encode(open(_oc).read().strip(), add_special_tokens=False)[:gen_len]
                _pad = getattr(ce.tokenizer, "eos_token_id", None)
                if _pad is None: _pad = (_ot[-1] if _ot else mid)
                code_ids = torch.tensor([pids + _ot + [_pad] * max(0, gen_len - len(_ot))], dtype=torch.long, device=device)
            _trace_on = bool(os.environ.get("AGENTICWAM_TRACE"))
            _trace = {"plen": int(plen), "gen_len": int(gen_len), "steps": []} if _trace_on else None
            _viz_on = bool(os.environ.get("AGENTICWAM_VIZ")) and not _VIZ.get("done")
            if _viz_on:
                _VIZ["steps"] = []; _VIZ["meta"] = {"plen": int(plen), "mask_id": int(mid)}
            eos = getattr(ce.tokenizer, "eos_token_id", None)
            latents_video = torch.randn(1, Cl, nc * Tlpv, Hl, Wl, device=device, dtype=cdt)
            for view in range(nc):
                d0, s0 = view * Tlpv, view * n_cond_lat
                latents_video[:, :, d0:d0 + n_cond_lat] = obs_latents[:, :, s0:s0 + n_cond_lat]
            latents_action = torch.randn(1, action_horizon, adim, device=device, dtype=cdt)
            self.video_scheduler.sigmas = self._vs_sigmas0.clone().to(device)
            self.video_scheduler.timesteps = self._vs_timesteps0.clone().to(device)
            self.video_scheduler.set_timesteps(num_inference_steps, device=device)
            ts_a, deltas_a = self.action_scheduler.build_inference_schedule(
                num_inference_steps, device, cdt, shift_override=sigma_shift)
            T = len(ts_a)
            # Compositional CFG (inference side; pairs with train-time AGENTICWAM_CFG_DROPOUT):
            #   pred = full + wc*(full - nocode) + wv*(full - novid)
            # Only branches the dropout actually trained: nocode = ALL-<MASK> code canvas
            # (the drop_code null; instruction prompt masked too, as in training), novid =
            # action's video attention severed. Both default 0 -> single-pass, byte-identical.
            _cfg_wc = float(os.environ.get("AGENTICWAM_CFG_WC", "0") or 0)
            _cfg_wv = float(os.environ.get("AGENTICWAM_CFG_WV", "0") or 0)
            code_pre_null = code_aux_null = None
            if _cfg_wc > 0:
                code_pre_null, _vP0 = self._pre_dit_vis(torch.full_like(code_ids, mid), _vit_pref)
                if _mem_K > 0 and _mem_M is not None:   # training's drop_code kept the injected memory prefix
                    _cn = code_pre_null["tokens"].clone()
                    _cn[:, _vP0:_vP0 + _mem_K, :] = self.code_memory.read(_mem_M).to(_cn.dtype)   # FIX 2026-09-03: after the ViT prefix
                    code_pre_null["tokens"] = _cn
                code_aux_null = {"t_mod": code_pre_null["t_mod"], "freqs": code_pre_null["freqs"], "context": None}
            if _cfg_wc > 0 or _cfg_wv > 0:
                logger.info("CFG-infer active: wc=%.2f wv=%.2f", _cfg_wc, _cfg_wv)
            for step_index, (t_a, d_a) in enumerate(zip(ts_a, deltas_a)):
                sigma_v = self.video_scheduler.sigmas[step_index].to(device=device, dtype=cdt)
                ts_v = self.video_scheduler.timesteps[step_index]
                v_sig_pf = torch.full((1, nc * Tlpv), float(sigma_v), device=device, dtype=cdt)
                cond_mask = torch.zeros(1, 1, nc * Tlpv, Hl, Wl, device=device, dtype=cdt)
                for view in range(nc):
                    d0 = view * Tlpv
                    v_sig_pf[:, d0:d0 + n_cond_lat] = self.cond_timestep
                    cond_mask[:, :, d0:d0 + n_cond_lat] = 1.0
                video_tokens, video_aux, shape_info = self._video_pre(latents_video, v_sig_pf, prompt_embs, cond_mask)
                ts_a_ = t_a.reshape(1).to(device=device, dtype=cdt)
                action_pre = self.action_expert.pre_dit(action_tokens=latents_action, timestep=ts_a_,
                                                        context=context, context_mask=context_mask)
                code_pre, _vitP = self._pre_dit_vis(code_ids, _vit_pref)
                if _mem_K > 0 and _mem_M is not None:
                    _ct = code_pre["tokens"].clone()
                    _ct[:, _vitP:_vitP + _mem_K, :] = self.code_memory.read(_mem_M).to(_ct.dtype)   # FIX 2026-09-03: after the ViT prefix
                    code_pre["tokens"] = _ct
                code_aux = {"t_mod": code_pre["t_mod"], "freqs": code_pre["freqs"], "context": None}
                tokens = {"video": video_tokens, "code": code_pre["tokens"], "action": action_pre["tokens"]}
                aux = {"video": video_aux, "code": code_aux,
                       "action": {"t_mod": action_pre["t_mod"], "freqs": action_pre["freqs"],
                                  "context": {"context": action_pre["context"], "mask": action_pre["context_mask"]}}}
                _vct = None
                _cpl = int(_vit_pref.shape[1]) if _vit_pref is not None else 0
                _cpx = (_cpl + _mem_K) if _cpl else 0     # sealed prefix = ViT rows + memory slots (slots are inside code_ids)
                mask = self._build_mot_attention_mask(video_tokens.shape[1], action_pre["tokens"].shape[1],
                                                      device, code_seq_len=code_ids.shape[1] + _cpl,
                                                      video_cond_tok=_vct, code_prefix_len=_cpx)
                mask_nv = None
                if _cfg_wv > 0:
                    self._cfg_mode = "drop_video_for_action"
                    mask_nv = self._build_mot_attention_mask(video_tokens.shape[1], action_pre["tokens"].shape[1],
                                                             device, code_seq_len=code_ids.shape[1] + _cpl, video_cond_tok=_vct, code_prefix_len=_cpx)
                    self._cfg_mode = "normal"
                out = self.mot(tokens, aux, mask)
                pred_video = self._video_post(out["video"], video_aux, shape_info)
                if not os.environ.get("AGENTICWAM_ABLATE_FREEZE_VIDEO"):
                    res = self.video_scheduler.step(pred_video, ts_v, latents_video, step_index)
                    latents_video = res[0] if isinstance(res, (tuple, list)) else res
                for view in range(nc):
                    d0, s0 = view * Tlpv, view * n_cond_lat
                    latents_video[:, :, d0:d0 + n_cond_lat] = obs_latents[:, :, s0:s0 + n_cond_lat]
                pred_action = self.action_expert.post_dit(out["action"], action_pre)
                if _cfg_wc > 0 or _cfg_wv > 0:
                    _pa_full = pred_action
                    if _cfg_wc > 0:
                        _out_nc = self.mot(
                            {"video": video_tokens, "code": code_pre_null["tokens"], "action": action_pre["tokens"]},
                            {"video": video_aux, "code": code_aux_null, "action": aux["action"]}, mask)
                        pred_action = pred_action + _cfg_wc * (
                            _pa_full - self.action_expert.post_dit(_out_nc["action"], action_pre))
                    if _cfg_wv > 0:
                        _out_nv = self.mot(tokens, aux, mask_nv)
                        pred_action = pred_action + _cfg_wv * (
                            _pa_full - self.action_expert.post_dit(_out_nv["action"], action_pre))
                latents_action = self.action_scheduler.step(pred_action, d_a, latents_action)
                if _viz_on:
                    _VIZ["steps"].append([int(x) for x in code_ids[0].tolist()])
                # ReMDM-style REVISION (env-gated, training-free): in the last R steps, re-mask the
                # lowest self-confidence ALREADY-REVEALED tokens so the model can undo committed
                # errors. Valid without retraining because training used a global mask ratio
                # t~U(0,1), so a partially-masked canvas is in-distribution. Scoring under the
                # (near-)full canvas is far more discriminative than confidence at reveal time.
                _rr = int(os.environ.get("AGENTICWAM_REMASK_ROUNDS", "0"))
                _remask_active = _rr > 0 and (T - 1 - _rr) <= step_index < (T - 1)
                if _remask_active:
                    _lg = ce.post_dit(out["code"])[0][_vitP:]   # drop ViT prefix rows: logits are [P+plen+gen], ids coords are [plen+gen]
                    _pp = _lg.float().softmax(-1)
                    _body = torch.arange(plen, code_ids.shape[1], device=device)
                    _cur = code_ids[0, _body]
                    _isrev = _cur != mid
                    if bool(_isrev.any()):
                        _pos = _body[_isrev]
                        _selfconf = _pp[_pos, _cur[_isrev]]
                        _fr = float(os.environ.get("AGENTICWAM_REMASK_FRAC", "0.20"))
                        _k = max(1, int(_fr * float(_pos.numel())))
                        _worst = torch.argsort(_selfconf)[:_k]
                        code_ids[0, _pos[_worst]] = mid
                masked = (code_ids[0, plen:] == mid).nonzero(as_tuple=True)[0] + plen
                if masked.numel() > 0 and not os.environ.get("AGENTICWAM_ABLATE_DROP_CODE"):
                    _pfx = int(os.environ.get("AGENTICWAM_PREFIX_LAST", "0"))       # right-to-left: HOLD the first _pfx code
                    rev = masked                                             # positions masked until the body commits,
                    if _pfx > 0 and step_index < T - 1:                      # so they condition on the full right-context
                        _nonpfx = masked[(masked - plen) >= _pfx]
                        if _nonpfx.numel() > 0:
                            rev = _nonpfx                                    # reveal body first; prefix falls out last
                    _thr = float(os.environ.get("AGENTICWAM_CONF_THRESH", "0"))
                    if _thr > 0:                                              # ADAPTIVE confidence-threshold decode:
                        logits = ce.post_dit(out["code"])[0][_vitP:]   # drop ViT prefix rows (ids coords)
                        # DIAGNOSTIC (env-gated): forbid EOS/pad on the first AGENTICWAM_MIN_CODE_LEN canvas
                        # positions, so an early high-confidence EOS cannot truncate the program to "".
                        _mnl = int(os.environ.get("AGENTICWAM_MIN_CODE_LEN", "0"))
                        if _mnl > 0:
                            _eid = getattr(ce.tokenizer, "eos_token_id", None)
                            _pid = getattr(ce.tokenizer, "pad_token_id", None)
                            _ban = [x for x in (_eid, _pid) if x is not None]
                            if _ban:
                                _early = rev[(rev - plen) < _mnl]
                                if _early.numel() > 0:
                                    for _b in _ban:
                                        logits[_early, _b] = float("-inf")
                        probs = logits[rev].float().softmax(-1)
                        # P0.1 STOCHASTIC DECODING (DiffuCoder-style, arXiv:2506.20639): temperature
                        # on BOTH entropy sources of masked-diffusion decode. Env-gated; both default
                        # 0 -> byte-identical argmax (eval unchanged). Needed because argmax gives a
                        # deterministic policy (measured pairwise Jaccard 0.9987 across seeds), which
                        # makes any group-based RL advantage exactly 0/0.
                        _ttemp = float(os.environ.get("AGENTICWAM_DECODE_TEMP", "0"))
                        if _ttemp > 0:
                            _pT = (logits[rev].float() / _ttemp).softmax(-1)
                            tok = torch.multinomial(_pT, 1).squeeze(-1)
                            conf = probs.gather(-1, tok.unsqueeze(-1)).squeeze(-1)  # rank by BASE prob
                        else:
                            conf, tok = probs.max(-1)
                        sel = conf > _thr
                        _cap = int(os.environ.get("AGENTICWAM_REVEAL_CAP", "9999"))
                        if _remask_active:
                            _cap = 10 ** 6      # revision pass: refill everything that clears tau
                        if int(sel.sum()) > _cap:
                            _otemp = float(os.environ.get("AGENTICWAM_ORDER_TEMP", "0"))
                            if _otemp > 0:   # Gumbel-perturbed reveal ORDER (top-k becomes sampling)
                                _g = -torch.log(-torch.log(torch.rand_like(conf).clamp_min(1e-9)).clamp_min(1e-9))
                                _score = torch.log(conf.clamp_min(1e-9)) + _otemp * _g
                                keep = torch.argsort(_score, descending=True)[:_cap]
                            else:
                                keep = torch.argsort(conf, descending=True)[:_cap]
                            m2 = torch.zeros_like(sel); m2[keep] = True; sel = sel & m2
                        if not bool(sel.any()) and step_index >= T - 3:
                            sel[int(torch.argmax(conf))] = True
                        if step_index == T - 1:
                            sel[:] = True
                        idx = sel.nonzero(as_tuple=True)[0]
                        if _trace_on:
                            _trace["steps"].append({
                                "step": int(step_index),
                                "n_masked_before": int(masked.numel()),
                                "revealed": [[int(rev[i]), int(tok[i]), round(float(conf[i]), 4)]
                                             for i in idx.tolist()],
                                "conf_all": [round(float(c), 4) for c in conf[:64].tolist()],
                            })
                        if idx.numel() > 0:
                            code_ids[0, rev[idx]] = tok[idx].to(code_ids.dtype)
                    else:
                        _wu = float(os.environ.get("AGENTICWAM_WARMUP", "0.0"))
                        _flf = float(os.environ.get("AGENTICWAM_FL_FRAC", "1.0"))
                        _frac = min(1.0, max(0.0, (step_index + 1 - _wu * T)) / max(1.0, _flf * T))
                        target = int(round(gen_len * _frac))
                        n_reveal = max(0, target - (gen_len - int(masked.numel())))
                        if step_index == T - 1:
                            n_reveal = int(masked.numel())
                        if n_reveal > 0:
                            logits = ce.post_dit(out["code"])[0][_vitP:]   # drop ViT prefix rows (ids coords)
                            k = min(int(n_reveal), int(masked.numel()))
                            if os.environ.get("AGENTICWAM_REVEAL_ORDER", "confidence") == "confidence":
                                probs = logits[masked].float().softmax(-1); conf, tok = probs.max(-1)
                                # EOS DEFERRAL (env-gated, default off): until `frac` of the canvas
                                # is committed, demote EOS-argmax positions to the back of the reveal
                                # order (rank only -- the token is untouched, and the final force-
                                # commit still writes EOS wherever it is argmax). Content gets first
                                # claim on positions; EOS settles into the tail.
                                _edf = float(os.environ.get("AGENTICWAM_EOS_DEFER_FRAC", "0") or 0)
                                if _edf > 0 and eos is not None and step_index < T - 1:
                                    _done = 1.0 - float(masked.numel()) / max(1.0, float(gen_len))
                                    if _done < _edf:
                                        conf = torch.where(tok == eos, conf - 1e6, conf)
                                order = torch.argsort(conf, descending=True)[:k]
                                code_ids[0, masked[order]] = tok[order].to(code_ids.dtype)
                                if os.environ.get("AGENTICWAM_DECODE_LOG"):
                                    import json as _dj
                                    with open(os.environ["AGENTICWAM_DECODE_LOG"], "a") as _df:
                                        _df.write(_dj.dumps({"step": int(step_index),
                                            "pos": [int(x) for x in masked[order].tolist()],
                                            "tok": [int(x) for x in tok[order].tolist()],
                                            "conf": [round(float(x), 4) for x in conf[order].tolist()]}) + "\n")
                            else:
                                tok = logits[masked].argmax(-1)
                                code_ids[0, masked[:k]] = tok[:k].to(code_ids.dtype)
            if _viz_on:
                _VIZ["steps"].append([int(x) for x in code_ids[0].tolist()])   # final state
                import json as _json
                try:
                    with open(os.environ["AGENTICWAM_VIZ"], "w") as _f:
                        _json.dump({"meta": _VIZ["meta"], "steps": _VIZ["steps"], "prompt": prompt}, _f)
                    _VIZ["done"] = True
                    logger.info("VIZ saved %d steps -> %s", len(_VIZ["steps"]), os.environ["AGENTICWAM_VIZ"])
                except Exception as _e:
                    logger.warning("viz save failed: %s", _e)
            action_out = latents_action[0].detach().float().cpu()
            gen = [t for t in code_ids[0, plen:].tolist() if t != mid]
            if eos is not None and eos in gen:
                if os.environ.get("AGENTICWAM_EOS_RUN_TRUNC", "0") == "1":
                    # terminal-run truncation: strip the trailing EOS run, drop isolated interior
                    # EOS -- a stray early EOS commit then costs one token, not the whole decoded
                    # suffix (first-EOS truncation chops everything after it)
                    while gen and gen[-1] == eos:
                        gen.pop()
                    gen = [t for t in gen if t != eos]
                else:
                    gen = gen[:gen.index(eos)]
            try:
                policy_code = ce.tokenizer.decode(gen, skip_special_tokens=True).strip()
            except Exception:
                policy_code = ""
            if _trace_on:
                import json as _tj
                _trace["final_ids"] = [int(x) for x in code_ids[0].tolist()]
                _trace["code"] = policy_code
                _trace["instruction"] = str(instruction)
                try:
                    with open(os.environ["AGENTICWAM_TRACE"] + "." + str(os.getpid()), "a") as _tf:
                        _tf.write(_tj.dumps(_trace) + "\n")
                except Exception:
                    pass
            if getattr(self, "code_memory_enabled", False) and policy_code:
                _prev = getattr(self, "_mem_code_history", "") or ""
                self._mem_code_history = (_prev + policy_code + "\n")[-4000:]   # rolling generated-history for the memory
            if return_video:
                gen_video = self._decode_video_latents(latents_video).detach().float().cpu()
                return action_out, policy_code, gen_video
            return action_out, policy_code

    @torch.no_grad()
    def _resolve_infer_code_tokens(self, explicit=None) -> int:
        """Code canvas length for inference. Precedence:
        AGENTICWAM_INFER_CODE_TOKENS env > explicit arg > config infer_code_tokens > training cap."""
        env = os.environ.get("AGENTICWAM_INFER_CODE_TOKENS")
        if env:
            return int(env)
        if explicit is not None:
            return int(explicit)
        if getattr(self, "infer_code_tokens", None):
            return int(self.infer_code_tokens)
        return int(getattr(self.code_expert, "max_code_tokens", 128))

    def infer_action_trimodal(self, obs_video, proprio, prompt_embs, action_horizon,
                              num_video_frames: int = 9, num_inference_steps: int = 20,
                              sigma_shift=None, instruction=None, policy_code=None,
                              code_cache=None, max_code_tokens: Optional[int] = None,
                              return_video: bool = False):
        """3-expert (code-in-middle) inference. (1) AR-generate the policy code conditioned on the
        obs (code->video joint attention), then (2) JOINTLY co-denoise the full future video +
        action with the FIXED code, the action attending to the full (per-step noised) video + the
        code — matching the code-in-middle training coupling (video autonomous -> code -> action).
        Returns (action[Ta,adim] fp32 cpu, policy_code[str][, gen_video])."""
        self.dit.eval()
        device = self.device
        cdt = next(self.action_expert.parameters()).dtype
        adim = self.action_expert.action_dim
        nc = self.n_cameras
        if self.code_expert is None:
            raise ValueError("infer_action_trimodal requires use_code_expert=true")
        max_code_tokens = self._resolve_infer_code_tokens(max_code_tokens)
        if getattr(self.code_expert, "code_objective", "") == "joint":
            return self._infer_action_joint(
                obs_video, proprio, prompt_embs, action_horizon,
                num_video_frames=num_video_frames, num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift, instruction=instruction,
                max_code_tokens=max_code_tokens, return_video=return_video)
        if proprio is not None and proprio.dim() == 2:
            proprio = proprio.unsqueeze(1)
        with torch.autocast(device_type="cuda", dtype=cdt,
                            enabled=cdt in (torch.bfloat16, torch.float16)):
            obs_latents = self._encode_video_latents(obs_video.to(device))
            _vit_pref = self._vit_prefix(obs_video.to(device))
            Cl, Hl, Wl = obs_latents.shape[1], obs_latents.shape[3], obs_latents.shape[4]
            n_cond_lat = obs_latents.shape[2] // nc
            Tlpv = max(self._pixel_to_latent(int(num_video_frames)), n_cond_lat)
            prompt_embs = prompt_embs.to(device=device, dtype=cdt)
            if proprio is not None:
                proprio = proprio.to(device=device, dtype=cdt)
            context, context_mask = self._action_context(prompt_embs, proprio)
            if instruction is None:
                instruction = ""

            # --- 1) code: AR-generate obs-conditioned (once; cache per-episode) ---
            if code_cache is not None and code_cache.get("code_pre") is not None:
                policy_code = code_cache["policy_code"]
                code_pre = code_cache["code_pre"]
            else:
                if policy_code is None:
                    n_obs = obs_latents.shape[2]
                    obs_sig = torch.full((1, n_obs), float(self.cond_timestep), device=device, dtype=cdt)
                    obs_cmask = torch.ones(1, 1, n_obs, Hl, Wl, device=device, dtype=cdt)
                    obs_vt, obs_va, _ = self._video_pre(obs_latents, obs_sig, prompt_embs, obs_cmask)
                    a_dummy = self.action_expert.pre_dit(
                        action_tokens=torch.zeros(1, 1, adim, device=device, dtype=cdt),
                        timestep=torch.zeros(1, device=device, dtype=cdt),
                        context=context, context_mask=context_mask)
                    policy_code = self._generate_code_obs(instruction, obs_vt, obs_va, a_dummy,
                                                          max_new_tokens=max_code_tokens)
                code_ids, _ = self.code_expert.build_code_batch([instruction], [policy_code], device=device)
                code_pre = self.code_expert.pre_dit(code_ids)
                if code_cache is not None:
                    code_cache["policy_code"] = policy_code
                    code_cache["code_pre"] = code_pre
            code_seq_len = int(code_pre["tokens"].shape[1])
            code_aux = {"t_mod": code_pre["t_mod"], "freqs": code_pre["freqs"], "context": None}

            # --- 2) joint video+action denoise with the fixed code ---
            latents_video = torch.randn(1, Cl, nc * Tlpv, Hl, Wl, device=device, dtype=cdt)
            for view in range(nc):
                d0, s0 = view * Tlpv, view * n_cond_lat
                latents_video[:, :, d0:d0 + n_cond_lat] = obs_latents[:, :, s0:s0 + n_cond_lat]
            latents_action = torch.randn(1, action_horizon, adim, device=device, dtype=cdt)
            self.video_scheduler.sigmas = self._vs_sigmas0.clone().to(device)
            self.video_scheduler.timesteps = self._vs_timesteps0.clone().to(device)
            self.video_scheduler.set_timesteps(num_inference_steps, device=device)
            ts_a, deltas_a = self.action_scheduler.build_inference_schedule(
                num_inference_steps, device, cdt, shift_override=sigma_shift)
            for step_index, (t_a, d_a) in enumerate(zip(ts_a, deltas_a)):
                sigma_v = self.video_scheduler.sigmas[step_index].to(device=device, dtype=cdt)
                ts_v = self.video_scheduler.timesteps[step_index]
                v_sig_pf = torch.full((1, nc * Tlpv), float(sigma_v), device=device, dtype=cdt)
                cond_mask = torch.zeros(1, 1, nc * Tlpv, Hl, Wl, device=device, dtype=cdt)
                for view in range(nc):
                    d0 = view * Tlpv
                    v_sig_pf[:, d0:d0 + n_cond_lat] = self.cond_timestep
                    cond_mask[:, :, d0:d0 + n_cond_lat] = 1.0
                video_tokens, video_aux, shape_info = self._video_pre(latents_video, v_sig_pf, prompt_embs, cond_mask)
                ts_a_ = t_a.reshape(1).to(device=device, dtype=cdt)
                action_pre = self.action_expert.pre_dit(action_tokens=latents_action, timestep=ts_a_,
                                                        context=context, context_mask=context_mask)
                tokens = {"video": video_tokens, "code": code_pre["tokens"], "action": action_pre["tokens"]}
                aux = {"video": video_aux, "code": code_aux,
                       "action": {"t_mod": action_pre["t_mod"], "freqs": action_pre["freqs"],
                                  "context": {"context": action_pre["context"], "mask": action_pre["context_mask"]}}}
                _vct = None
                _cpl = int(_vit_pref.shape[1]) if "_vit_pref" in dir() and _vit_pref is not None else 0
                mask = self._build_mot_attention_mask(video_tokens.shape[1], action_pre["tokens"].shape[1],
                                                      device, code_seq_len + _cpl, video_cond_tok=_vct,
                                                      code_prefix_len=_cpl)
                out = self.mot(tokens, aux, mask)
                pred_video = self._video_post(out["video"], video_aux, shape_info)
                if not os.environ.get("AGENTICWAM_ABLATE_FREEZE_VIDEO"):
                    res = self.video_scheduler.step(pred_video, ts_v, latents_video, step_index)
                    latents_video = res[0] if isinstance(res, (tuple, list)) else res
                for view in range(nc):
                    d0, s0 = view * Tlpv, view * n_cond_lat
                    latents_video[:, :, d0:d0 + n_cond_lat] = obs_latents[:, :, s0:s0 + n_cond_lat]
                pred_action = self.action_expert.post_dit(out["action"], action_pre)
                latents_action = self.action_scheduler.step(pred_action, d_a, latents_action)
            action_out = latents_action[0].detach().float().cpu()
            if return_video:
                gen_video = self._decode_video_latents(latents_video).detach().float().cpu()
                return action_out, policy_code, gen_video
            return action_out, policy_code

    def forward(self, sample, **kw):
        return self.training_loss(sample)
