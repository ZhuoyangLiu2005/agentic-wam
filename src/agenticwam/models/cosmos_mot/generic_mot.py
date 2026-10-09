"""Generic mixed-attention MoT coupler for heterogeneous experts.

Rather than hardcoding one block layout (`self_attn.q/k/v`, a static `modulation` param,
ungated cross-attn), each expert here
implements `pre_attn` / `post_attn` so its OWN block computation is preserved exactly.
This is required because the Cosmos-2.5-2B video expert uses AdaLN-**Zero** (modulation
derived from the diffusion timestep, and a *gated* cross-attn), which cannot be expressed
through the Wan post-block. The MoT itself only performs the cross-expert *joint
self-attention* (concatenate q/k/v across experts, one masked attention, split).

Expert contract (all experts must share num_layers / num_heads / attn_head_dim):
    expert.blocks: list[nn.Module]                # len == num_layers
    expert.num_heads: int
    expert.attn_head_dim: int
    expert.use_gradient_checkpointing: bool
    expert.pre_attn(block, x, aux, layer_idx) -> (q, k, v, post_state)
        # q/k/v: [B, S, num_heads*attn_head_dim], with the expert's own qk-norm + RoPE applied
    expert.post_attn(block, post_state, mixed_out, aux, layer_idx) -> x   # [B, S, D]
        # applies the attn out-projection to `mixed_out`, then the expert's cross-attn + FFN

Target config for this project (from Cosmos-2.5-2B): num_layers=28, num_heads=16,
attn_head_dim=128. Video expert = Cosmos DiT; code expert = Qwen3-VL-2B (28L, 2048 hidden);
action expert = ActionDiT rebuilt at 28/16/128.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..experts.wan_video_dit import flash_attention
from agenticwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class GenericMoT(nn.Module):
    def __init__(self, experts: Dict[str, nn.Module], checkpoint_mixed_attn: bool = True):
        super().__init__()
        if not experts:
            raise ValueError("`experts` cannot be empty.")
        if "video" not in experts:
            raise ValueError("`experts` must include a 'video' expert.")

        self.experts = nn.ModuleDict(experts)
        self.expert_order = list(self.experts.keys())
        self.checkpoint_mixed_attn = checkpoint_mixed_attn

        e0 = self.experts[self.expert_order[0]]
        self.num_layers = len(e0.blocks)
        self.num_heads = e0.num_heads
        self.attn_head_dim = e0.attn_head_dim
        for name in self.expert_order[1:]:
            e = self.experts[name]
            if len(e.blocks) != self.num_layers:
                raise ValueError(f"{name}: num_layers {len(e.blocks)} != {self.num_layers}")
            if e.num_heads != self.num_heads:
                raise ValueError(f"{name}: num_heads {e.num_heads} != {self.num_heads}")
            if e.attn_head_dim != self.attn_head_dim:
                raise ValueError(f"{name}: attn_head_dim {e.attn_head_dim} != {self.attn_head_dim}")

        logger.info(
            f"GenericMoT experts={self.expert_order} num_layers={self.num_layers} "
            f"num_heads={self.num_heads} head_dim={self.attn_head_dim}"
        )
        for name in self.expert_order:
            n = sum(p.numel() for p in self.experts[name].parameters()) / 1e9
            logger.info(f"  Expert '{name}': num_params={n:.2f} B")

    def _mixed_attention(self, q, k, v, attention_mask):
        def _fwd(q, k, v):
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attention_mask)

        if self.checkpoint_mixed_attn and self.training:
            return checkpoint(_fwd, q, k, v, use_reentrant=False)
        return _fwd(q, k, v)

    def forward(
        self,
        tokens: Dict[str, torch.Tensor],
        aux: Dict[str, Dict[str, Any]],
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """One tri-modal MoT forward.

        Args:
            tokens: {expert_name: [B, S_name, D]} input tokens per expert.
            aux:    {expert_name: dict} per-expert conditioning (timestep/rope/text/masks);
                    consumed by that expert's pre_attn/post_attn.
            attention_mask: joint [S_total, S_total] boolean mask over the concatenated
                            expert token order (self.expert_order).
        Returns:
            {expert_name: [B, S_name, D]} updated tokens.
        """
        missing = [k for k in self.expert_order if k not in tokens]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")

        tokens = dict(tokens)
        for layer_idx in range(self.num_layers):
            q_chunks, k_chunks, v_chunks, states, seq_lens = [], [], [], {}, []
            for name in self.expert_order:
                expert = self.experts[name]
                block = expert.blocks[layer_idx]
                q, k, v, st = expert.pre_attn(block, tokens[name], aux[name], layer_idx)
                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(q.shape[1])
                states[name] = st

            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)
            total = q_cat.shape[1]
            if attention_mask.shape[-1] != total:
                raise ValueError(
                    f"attention_mask seq {tuple(attention_mask.shape)} != total tokens {total}"
                )

            mixed = self._mixed_attention(q_cat, k_cat, v_cat, attention_mask)

            start = 0
            for name, slen in zip(self.expert_order, seq_lens):
                expert = self.experts[name]
                block = expert.blocks[layer_idx]
                tokens[name] = expert.post_attn(
                    block, states[name], mixed[:, start:start + slen, :], aux[name], layer_idx
                )
                start += slen

        return tokens
