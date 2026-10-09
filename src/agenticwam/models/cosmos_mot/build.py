"""Factory: construct a `AgenticWAMCosmos` tri-modal model and load the video-pretrained checkpoint.

The Cosmos DiT / VAE / VideoTokenizer / text-encoder structures mirror the
video-pretrained checkpoint, so its safetensors shards load key-for-key (config.json:
model_size=2B, view_embed_dim=7, max_n_cameras=4, z_dim=48, sigma_shift=5.0, Cosmos-Reason1).
The action/code experts are NOT in the video-only checkpoint (trained from scratch), so their
ffn/freq dims only need internal consistency at 28 layers / 16 heads / 128 head_dim.
"""
from __future__ import annotations

import dataclasses
import glob
import os
from typing import Optional

import torch
import logging
logger = logging.getLogger(__name__)
import torch.nn as nn

from ..cosmos import (
    COSMOS_2B_CONFIG,
    CosmosTransformer3DModel,
    CosmosTransformerConfig,
    FlowMatchUniPCScheduler,
    MultiViewRoPE,
    SchedulerConfig,
    VideoTokenizer,
)
from ..experts.action_dit import ActionDiT  # noqa: F401  (imported for reference / typing)
from ..experts.code_expert import build_code_expert
from ..experts.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .cosmos_video_expert import CosmosVideoExpert
from .wan_experts import WanActionExpert, WanCodeExpert
from .agenticwam_model import AgenticWAMCosmos


def build_cosmos_dit_config(
    *, latent_channels: int = 48, view_embed_dim: int = 7, max_n_cameras: int = 4
) -> CosmosTransformerConfig:
    """COSMOS_2B_CONFIG overridden for the 48-ch Wan2.2-VAE robotics backbone.

    in_channels = latent(48) + cond_mask(1) = 49 (the VideoTokenizer adds +1 pad + view_embed).
    """
    return dataclasses.replace(
        COSMOS_2B_CONFIG,
        in_channels=latent_channels + 1,   # 49
        out_channels=latent_channels,      # 48
        view_embed_dim=view_embed_dim,     # 7
        max_n_cameras=max_n_cameras,       # 4
    )


def build_agenticwam_cosmos(
    *,
    action_dim: int = 64,
    proprio_dim: int = 0,
    n_cameras: int = 4,
    action_hidden_dim: int = 1024,   # action expert residual width (DECOUPLED from the 16x128=2048
                                     # joint-attn head space); 1024 -> ~0.72B (matches AgenticWAM), vs
                                     # 2048 -> 1.92B. qkv project hidden->2048, output 2048->hidden.
    action_ffn_dim: int = 4096,
    action_freq_dim: int = 256,
    eps: float = 1e-6,
    code_model_name: str = "mock",   # "mock" for smoke; Qwen3-VL-2B path for real
    code_objective: str = "ar",           # "ar" (next-token) | "blockdiff" (SDAR block-diffusion)
    code_block_length: int = 4,
    max_code_tokens: int = 512,           # cap on (instruction prompt + code) tokens; batches pad
    code_canvas: int = 0,                 # TRAINING fixed code canvas (0 = batch-max alignment, legacy)
    infer_code_tokens: Optional[int] = None,   # INFERENCE-only code canvas (code tokens only); None -> max_code_tokens
                                          # to the batch max capped by this, so it is a truncation
                                          # bound, not a fixed canvas.
    text_encoder_path: str = "/path/to/Cosmos-Reason1-7B",
    vae_path: str = "/path/to/Wan2.2_VAE.pth",
    use_code_expert: bool = True,
    include_text_encoder: bool = False,   # False: use cached prompt_embs, keep 8.3B out of training
    video_noise_beta=None,                # (alpha,beta,noise_s) e.g. (7.0,1.0,0.999); None=uniform
    num_cond_frames: int = 0,             # obs frames/view kept clean (video prediction); 0=pure generation
    use_gradient_checkpointing: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    sigma_shift: float = 5.0,
) -> AgenticWAMCosmos:
    cfg = build_cosmos_dit_config()
    hidden = cfg.num_attention_heads * cfg.attention_head_dim  # 2048

    # video expert (Cosmos DiT) + tokenizer + multiview rope
    dit = CosmosTransformer3DModel(cfg)
    video_expert = CosmosVideoExpert(dit, use_gradient_checkpointing=use_gradient_checkpointing)
    video_tokenizer = VideoTokenizer(cfg)
    video_rope = MultiViewRoPE(
        cfg.attention_head_dim, cfg.max_size, cfg.patch_size, rope_scale=cfg.rope_scale
    )

    # frozen backbones (weights come from the video checkpoint)
    vae = _build_vae(vae_path)
    text_encoder = _build_text_encoder(text_encoder_path) if include_text_encoder else None

    # action expert (Wan-style, from scratch): 28 layers, 16x128 joint-attn width, but a NARROWER
    # residual stream (action_hidden_dim, default 1024) -> ~0.72B. The qkv project hidden->2048 so
    # it still couples with the 2048-wide Cosmos video expert in the MoT joint self-attention.
    action_expert = WanActionExpert(
        hidden_dim=action_hidden_dim,
        action_dim=action_dim,
        ffn_dim=action_ffn_dim,
        text_dim=cfg.text_embed_dim,   # 1024 (projected Reason1 text)
        freq_dim=action_freq_dim,
        eps=eps,
        num_heads=cfg.num_attention_heads,
        attn_head_dim=cfg.attention_head_dim,
        num_layers=cfg.num_layers,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    # code expert (Qwen3-VL-2B, 28 layers) — reassign class to add the MoT mixin
    code_expert = None
    if use_code_expert:
        code_expert = build_code_expert(
            model_name=code_model_name,
            num_layers=cfg.num_layers,           # 28
            num_heads=cfg.num_attention_heads,   # 16
            attn_head_dim=cfg.attention_head_dim,  # 128
            n_extra_layers=0,
            hidden_dim=hidden,
            objective=code_objective,
            block_length=code_block_length,
            max_code_tokens=max_code_tokens,
            code_canvas=code_canvas,
            device=device,
            torch_dtype=dtype,
        )
        code_expert.__class__ = WanCodeExpert
        if not hasattr(code_expert, "use_gradient_checkpointing"):
            code_expert.use_gradient_checkpointing = use_gradient_checkpointing

    video_scheduler = FlowMatchUniPCScheduler(
        SchedulerConfig(num_train_timesteps=1000, shift=sigma_shift)
    )
    action_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000)

    model = AgenticWAMCosmos(
        video_expert=video_expert,
        action_expert=action_expert,
        code_expert=code_expert,
        vae=vae,
        video_tokenizer=video_tokenizer,
        video_rope=video_rope,
        text_encoder=text_encoder,
        video_scheduler=video_scheduler,
        action_scheduler=action_scheduler,
        proprio_dim=proprio_dim,
        text_proj_dim=cfg.text_embed_dim,
        n_cameras=n_cameras,
        num_cond_frames=num_cond_frames,
        video_sigma_shift=sigma_shift,
        video_noise_beta=video_noise_beta,
        use_code_expert=use_code_expert,
        infer_code_tokens=infer_code_tokens,
    )
    # Keep fp32 (the trainer runs under accelerator.autocast → bf16 compute). A blanket .to(bf16)
    # would corrupt ActionDiT's complex RoPE freqs. Only move device here.
    return model.to(device=device)


def load_wan_vae(model: AgenticWAMCosmos, vae_path: str):
    """Load the frozen Wan2.2 VAE from Wan-AI's public `Wan2.2_VAE.pth` (Hugging Face Wan-AI/Wan2.2-TI2V-5B).

    The video pretraining stored this VAE in bf16; casting the public fp32 weights to bf16 reproduces it
    bit for bit. AgenticWAM checkpoints never contain the VAE, so every build loads it here."""
    if not os.path.isfile(vae_path):
        raise FileNotFoundError(f"Wan2.2 VAE not found: {vae_path} (paths.wan_vae, docs/INSTALL.md section 3)")
    sd = torch.load(vae_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    remap = {"vae.model." + k: v.to(torch.bfloat16) for k, v in sd.items()}
    want = {k for k in model.state_dict() if k.startswith("vae.")}
    if want - set(remap):
        raise RuntimeError(f"{vae_path}: {len(want - set(remap))} VAE tensors missing, e.g. {sorted(want - set(remap))[:3]}")
    model.load_state_dict(remap, strict=False)
    for p in model.vae.parameters():
        p.requires_grad_(False)
    logger.info(f"Wan2.2 VAE loaded from {vae_path}: {len(want)} tensors")


def load_video_pretrain(model: AgenticWAMCosmos, ckpt_dir: str, strict: bool = False):
    """Load the 5-shard safetensors video checkpoint into the model."""
    from safetensors.torch import load_file

    sd = {}
    shards = sorted(glob.glob(os.path.join(ckpt_dir, "model-*-of-*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards in {ckpt_dir}")
    for s in shards:
        sd.update(load_file(s))
    res = model.load_cosmos_video_checkpoint(sd, strict=strict)
    # re-freeze frozen backbones (assign-load can reset requires_grad)
    for p in model.vae.parameters():
        p.requires_grad_(False)
    if model.text_encoder is not None:
        for p in model.text_encoder.parameters():
            p.requires_grad_(False)
    return res


def create_agenticwam_cosmos(
    *,
    model_dtype=torch.bfloat16,
    device: str = "cuda",
    checkpoint_dir: Optional[str] = None,
    action_dim: int = 64,
    proprio_dim: int = 64,
    n_cameras: int = 4,
    action_hidden_dim: int = 1024,   # ~0.72B action expert (matches AgenticWAM); decoupled from 2048 attn
    action_ffn_dim: int = 4096,
    use_code_expert: bool = False,
    code_model_name: str = "mock",
    code_objective: str = "ar",
    code_block_length: int = 4,
    max_code_tokens: int = 512,
    code_canvas: int = 0,                      # training fixed code canvas; 0 = batch-max (legacy)
    infer_code_tokens: Optional[int] = None,   # inference-only code canvas; None -> max_code_tokens
    code_vision: str = "none",               # "none" | "qwen3vl_prefix" (frozen ViT + trainable merger -> 210-token code prefix)
    code_vision_path: str = "/path/to/Qwen3-VL-2B-Instruct",
    include_text_encoder: bool = False,
    video_noise_beta=(7.0, 1.0, 0.999),
    num_cond_frames: int = 1,             # condition on the first obs frame/view; predict the future
    sigma_shift: float = 5.0,
    loss_lambda_video: float = 1.0,
    loss_lambda_action: float = 1.0,
    loss_lambda_code: float = 0.5,
    # --- objective-specific sigma weighting (default OFF = byte-identical training) ---
    sigma_reweight: bool = False,          # enable importance-weighted per-objective sigma
    sigma_reweight_alpha: float = 0.5,     # mixture weight on the CURRENT (video) density
    sigma_reweight_clip: float = 4.0,      # max importance weight after clipping
    sigma_reweight_normalize: bool = True, # renormalise weights to mean 1 per batch
    sigma_reweight_stratified: bool = True,# fixed split per batch instead of i.i.d. draws
    code_memory: bool = False,             # E3: learned recurrent code memory (plan across replans)
    code_memory_slots: int = 32,
    code_memory_heads: int = 8,
    code_memory_selfgen_p: float = 0.5,    # scheduled-sampling prob: encode SELF-predicted history

    load_pretrain: bool = False,
    wan_vae_path: Optional[str] = None,
    use_gradient_checkpointing: bool = True,
    **_ignored,
) -> AgenticWAMCosmos:
    """Hydra `_target_` entrypoint: build AgenticWAMCosmos + load the video-pretrained checkpoint.

    Called by `runtime.run_training` as `instantiate(cfg.model, model_dtype=, device=)`.
    Defaults to the training configuration (no in-model text encoder → cached prompt_embs;
    Beta(7,1) video sigma; fp32 params for autocast training).
    """
    model = build_agenticwam_cosmos(
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        n_cameras=n_cameras,
        action_hidden_dim=action_hidden_dim,
        action_ffn_dim=action_ffn_dim,
        code_model_name=code_model_name,
        code_objective=code_objective,
        code_block_length=code_block_length,
        max_code_tokens=max_code_tokens,
        code_canvas=code_canvas,
        infer_code_tokens=infer_code_tokens,
        use_code_expert=use_code_expert,
        include_text_encoder=include_text_encoder,
        video_noise_beta=(tuple(video_noise_beta) if video_noise_beta else None),
        num_cond_frames=num_cond_frames,
        sigma_shift=sigma_shift,
        use_gradient_checkpointing=use_gradient_checkpointing,
        device=device,
        dtype=model_dtype,
    )
    model.loss_lambda_video = loss_lambda_video
    model.loss_lambda_action = loss_lambda_action
    model.loss_lambda_code = loss_lambda_code
    # E3 learned code memory: construct as a submodule BEFORE the optimizer is built so its
    # params train. Disabled -> None (byte-identical). dim = code stream width (2048).
    model.code_memory_enabled = bool(code_memory)
    model.code_memory_selfgen_p = float(code_memory_selfgen_p)
    if code_memory and model.code_expert is not None:
        from ..experts.code_memory import LearnedCodeMemory
        _md = int(getattr(model.code_expert, "hidden_size", 0) or getattr(model.code_expert, "dim", 0) or 2048)
        _dt = next(model.parameters()).dtype if any(True for _ in model.parameters()) else None
        model.code_memory = LearnedCodeMemory(dim=_md, num_slots=int(code_memory_slots),
                                              num_heads=int(code_memory_heads)).to(device=device, dtype=_dt)
    else:
        model.code_memory = None
    model.sigma_reweight = bool(sigma_reweight)
    model.sigma_reweight_alpha = float(sigma_reweight_alpha)
    model.sigma_reweight_clip = float(sigma_reweight_clip)
    model.sigma_reweight_normalize = bool(sigma_reweight_normalize)
    model.sigma_reweight_stratified = bool(sigma_reweight_stratified)
    if load_pretrain:              # pretraining from scratch: video expert + tokenizer from a Cosmos video checkpoint
        load_video_pretrain(model, checkpoint_dir)
    if wan_vae_path:               # the frozen VAE is never part of an AgenticWAM checkpoint
        load_wan_vae(model, wan_vae_path)
    model.code_vision = None
    if str(code_vision) == "qwen3vl_prefix" and use_code_expert:   # the vision prefix only feeds the code expert
        from .code_vision import CodeVisionPrefix
        model.code_vision = CodeVisionPrefix(code_vision_path, dtype=model_dtype).to(device)
        print(f"[build] code_vision=qwen3vl_prefix attached (out_dim={model.code_vision.out_dim})", flush=True)
    return model


# ── frozen-backbone constructors (mirror the video checkpoint) ────────────────
def _build_vae(vae_path: str):
    """Wan2.2 VAE (WanVideoVAE38: dim=160, z=48, asymmetric dec_dim=256, 48-value scaling).

    This is the VAE the video pretraining used, NOT the Wan2.1
    `components/wan_video_vae.py`. Structure only — weights come from the checkpoint's `vae.*`.
    """
    from ..cosmos.wan22_video_vae import WanVideoVAE38
    return WanVideoVAE38(z_dim=48, dim=160)


class _TextEncoderWrapper(nn.Module):
    """Cosmos-Reason1 (Qwen2.5-VL) text encoder: concat of the 28 normalized layer hidden-states
    -> [B, N, 100352] (the crossattn_proj input). Built empty from HF config so the checkpoint's
    `text_encoder.text_encoder.*` (729 keys) loads without a 15GB base download."""

    def __init__(self, hf_config_id: str = "nvidia/Cosmos-Reason1-7B"):
        super().__init__()
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoConfig
        try:
            from accelerate import init_empty_weights
            with init_empty_weights():
                self.text_encoder = Qwen2_5_VLForConditionalGeneration(AutoConfig.from_pretrained(hf_config_id))
        except Exception:
            self.text_encoder = Qwen2_5_VLForConditionalGeneration(AutoConfig.from_pretrained(hf_config_id))

    @torch.no_grad()
    def forward(self, input_ids):
        hs = self.text_encoder(input_ids, output_hidden_states=True).hidden_states
        return torch.cat([(h - h.mean(-1, keepdim=True)) / (h.std(-1, keepdim=True) + 1e-8)
                          for h in hs[1:]], dim=-1)


def _build_text_encoder(text_encoder_path: str):
    # text_encoder_path unused: structure is built from HF config; weights load from the checkpoint.
    return _TextEncoderWrapper()
