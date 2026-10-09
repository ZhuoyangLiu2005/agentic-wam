"""Cosmos-2.5-2B DiT as a `GenericMoT` video expert.

Wraps the byte-faithful `CosmosTransformer3DModel` (so the checkpoint's `transformer.*`
checkpoint weights load key-for-key) and exposes `pre_attn` / `post_attn` that reproduce
the Cosmos block EXACTLY:

    norm_hs, gate1 = norm1(x, ts, temb);  x = x + gate1 * attn1(norm_hs, rope)      # self
    norm_hs, gate2 = norm2(x, ts, temb);  x = x + gate2 * attn2(norm_hs, text, ...) # cross (MultiView)
    norm_hs, gate3 = norm3(x, ts, temb);  x = x + gate3 * ff(norm_hs)               # ffn

The self-attention is decomposed for the MoT: `pre_attn` runs norm1 + to_q/to_k/to_v +
per-head qk-norm + Cosmos 3D RoPE and returns q/k/v (the MoT does the *joint* attention
across experts); `post_attn` applies attn1.to_out + the cross-attn + FFN with the AdaLN-Zero
gates. This is the Cosmos analogue of AgenticWAM's Wan-block pre/post — the pretrained weights load
unchanged and the video branch's math matches the video pretraining.

RoPE reconciliation: Cosmos `apply_rotary_emb` operates on [B,H,S,Dh]; AgenticWAM's
`flash_attention` reshapes `(b s (n d) -> b n s d)`. We apply norm_q/norm_k + RoPE on
[B,H,S,Dh] then reshape back to [B,S,H*Dh] — the head layout matches, so the MoT's joint
attention sees correctly-rotated video q/k concatenated with the (separately-rotated)
action/code q/k.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from ..cosmos.transformer_cosmos import apply_rotary_emb


class CosmosVideoExpert(nn.Module):
    def __init__(self, cosmos_transformer: nn.Module, use_gradient_checkpointing: bool = True):
        super().__init__()
        self.dit = cosmos_transformer  # CosmosTransformer3DModel: transformer_blocks + patch/pos/time/norm_out
        self.blocks = cosmos_transformer.transformer_blocks
        cfg = cosmos_transformer.config
        self.num_heads = cfg.num_attention_heads      # 16
        self.attn_head_dim = cfg.attention_head_dim   # 128
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def pre_attn(self, block, x, aux: Dict[str, Any], layer_idx: int):
        ts = aux["embedded_timestep"]
        temb = aux.get("temb")
        rope = aux.get("rope")
        extra = aux.get("extra_pos_emb")
        if extra is not None:
            x = x + extra  # Cosmos adds extra_pos_emb at the start of each block

        H, Dh = self.num_heads, self.attn_head_dim
        norm_hs, gate1 = block.norm1(x, ts, temb)
        q = block.attn1.to_q(norm_hs)
        k = block.attn1.to_k(norm_hs)
        v = block.attn1.to_v(norm_hs)
        B, S, _ = q.shape
        q = q.view(B, S, H, Dh).transpose(1, 2)  # [B,H,S,Dh]
        k = k.view(B, S, H, Dh).transpose(1, 2)
        v = v.view(B, S, H, Dh).transpose(1, 2)
        q = block.attn1.norm_q(q)
        k = block.attn1.norm_k(k)
        if rope is not None:
            q = apply_rotary_emb(q, rope)
            k = apply_rotary_emb(k, rope)
        q = q.transpose(1, 2).reshape(B, S, H * Dh)  # back to MoT layout
        k = k.transpose(1, 2).reshape(B, S, H * Dh)
        v = v.transpose(1, 2).reshape(B, S, H * Dh)
        return q, k, v, {"x": x, "gate1": gate1, "ts": ts, "temb": temb}

    def post_attn(self, block, st, mixed_out, aux: Dict[str, Any], layer_idx: int):
        # self-attn residual (attn out-projection; to_out[1] is Identity)
        x = st["x"] + st["gate1"] * block.attn1.to_out[0](mixed_out)
        ts, temb = st["ts"], st["temb"]
        # cross-attn to text (MultiView), gated
        norm_hs, gate2 = block.norm2(x, ts, temb)
        x = x + gate2 * block.attn2(
            norm_hs,
            encoder_hidden_states=aux.get("text_ctx"),
            n_cameras=aux.get("n_cameras", 1),
            video_idx=aux.get("video_idx"),
            non_video_idx=aux.get("non_video_idx"),
        )
        # feed-forward, gated
        norm_hs, gate3 = block.norm3(x, ts, temb)
        x = x + gate3 * block.ff(norm_hs)
        return x
