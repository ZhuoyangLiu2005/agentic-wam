"""Wan-style experts (action, code) for the GenericMoT.

AgenticWAM's ActionDiT and CodeExpert already expose the Wan block interface
(`block.modulation` static param, `block.norm1/2/3`, `block.self_attn.{q,k,v,norm_q,norm_k,o}`,
`block.cross_attn` (or None), `block.ffn`, `block.gate`, `block.num_heads`). This mixin adds
the `pre_attn`/`post_attn` the GenericMoT calls — it is exactly AgenticWAM `mot.py`'s
`_build_expert_attention_io` + `_apply_expert_post_block` + `_split_modulation`, refactored to
instance methods so the Cosmos video expert (AdaLN-Zero) and these Wan-style experts can be
mixed in one MoT. Instantiate both at 28 layers / 16 heads / 128 head_dim to match Cosmos.

aux contract (per Wan expert):
    aux["t_mod"]  : [B,(S,)6*D] time modulation (video/action diffusion) or 0 (code = neutral)
    aux["freqs"]  : RoPE freqs [S,1,rope_dim] for this expert
    aux["context"]: optional {"context": [B,L,D], "mask": [B,S,L] or [B,1,S,L]} for cross-attn
                    (action = text+proprio; code = None — code has no cross-attn)
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from ..experts.wan_video_dit import modulate, rope_apply
from ..experts.action_dit import ActionDiT
from ..experts.code_expert import CodeExpert


class WanExpertMixin:
    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        parts = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            parts = tuple(p.squeeze(2) for p in parts)
        return parts  # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def pre_attn(self, block, x, aux: Dict[str, Any], layer_idx: int):
        t_mod = aux["t_mod"]
        freqs = aux["freqs"]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)
        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)
        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)
        st = {
            "x": x,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
        }
        return q, k, v, st

    def post_attn(self, block, st, mixed_out, aux: Dict[str, Any], layer_idx: int):
        x = block.gate(st["x"], st["gate_msa"], block.self_attn.o(mixed_out))
        context = aux.get("context")
        if context is not None and getattr(block, "cross_attn", None) is not None:
            ctx = context.get("context")
            if ctx is not None:
                mask = context.get("mask")
                if mask is not None and mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), ctx, ctx_mask=mask)
        mlp_input = modulate(block.norm2(x), st["shift_mlp"], st["scale_mlp"])
        x = block.gate(x, st["gate_mlp"], block.ffn(mlp_input))
        return x


class WanActionExpert(WanExpertMixin, ActionDiT):
    """ActionDiT (Wan-style, with text+proprio cross-attn) as a GenericMoT expert.
    Instantiate at num_layers=28, num_heads=16, attn_head_dim=128 to match Cosmos."""
    pass


class WanCodeExpert(WanExpertMixin, CodeExpert):
    """Qwen3-VL-2B code expert (Wan-style, no cross-attn, neutral modulation) as a GenericMoT
    expert. Use 28 Qwen layers (drop AgenticWAM's +2 padding) at 16 heads / 128 head_dim.
    Keeps CodeExpert's pre_dit/post_dit/code_loss/generate for the AR code stream."""
    pass
