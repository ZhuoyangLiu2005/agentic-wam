"""Learned recurrent code memory for the AGENTICWAM-Cosmos joint code expert (E3).

A fixed-size K-slot memory that summarizes the policy-code generated so far in an episode
and is READ as a conditioning prefix by the code expert each replan — giving the code a
stable GLOBAL PLAN across replans instead of a re-derived local slice.

Design (RMT / Perceiver-latent recurrent memory):
  - K learned slots M[B,K,d]; per-replan gated cross-attention WRITE from the new code
    hidden states, CONFIDENCE-GATED so low-confidence garble contributes less to the plan;
  - READ returns LayerNorm(M) as K prefix tokens for the code stream.
Stability: gated residual write (bounded), LayerNorms, small gate-bias init (slow write),
per-slot positional embedding (prevents slot collapse). The module is only instantiated
when the memory path is enabled at the call site; disabled -> nothing changes.
"""
from typing import Optional
import torch
import torch.nn as nn


class LearnedCodeMemory(nn.Module):
    def __init__(self, dim: int = 2048, num_slots: int = 32, num_heads: int = 8,
                 gate_bias: float = -2.0, ffn_mult: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim, self.K, self.H = dim, num_slots, num_heads
        self.slot_init = nn.Parameter(torch.randn(num_slots, dim) * 0.02)
        self.slot_pos = nn.Parameter(torch.randn(num_slots, dim) * 0.02)   # anti-collapse
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim); self.o = nn.Linear(dim, dim)
        self.kv_ln = nn.LayerNorm(dim); self.write_ln = nn.LayerNorm(dim)
        self.gate = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, gate_bias)  # slow start
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_mult * dim), nn.GELU(),
                                 nn.Linear(ffn_mult * dim, dim))
        self.ffn_ln = nn.LayerNorm(dim)
        self.state_ln = nn.LayerNorm(dim)   # normalize recurrent state -> bounded ||M||
        self.read_ln = nn.LayerNorm(dim)

    def init_state(self, batch: int, device=None, dtype=None) -> torch.Tensor:
        M = (self.slot_init + self.slot_pos).unsqueeze(0).expand(batch, -1, -1).contiguous()
        if device is not None or dtype is not None:
            M = M.to(device=device, dtype=dtype)
        return M

    def _attend(self, M, ctx, conf=None, valid=None):
        B, K, d = M.shape; L = ctx.shape[1]; H, hd = self.H, self.dim // self.H
        q = self.q(M).view(B, K, H, hd).transpose(1, 2)
        k = self.k(ctx).view(B, L, H, hd).transpose(1, 2)
        v = self.v(ctx).view(B, L, H, hd).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / (hd ** 0.5)          # [B,H,K,L]
        bias = None
        if conf is not None:   # confidence as an additive log-weight on the KV
            bias = torch.log(conf.clamp(1e-4, 1.0))[:, None, None, :]
        if valid is not None:
            vb = torch.where(valid, 0.0, float("-inf"))[:, None, None, :]
            bias = vb if bias is None else bias + vb
        if bias is not None:
            att = att + bias
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, K, d)
        return self.o(out)

    def write(self, M: torch.Tensor, code_hidden: torch.Tensor,
              code_conf: Optional[torch.Tensor] = None,
              valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """M:[B,K,d] code_hidden:[B,L,d] code_conf:[B,L]in[0,1] valid_mask:[B,L]bool"""
        u = self.write_ln(self._attend(M, self.kv_ln(code_hidden), code_conf, valid_mask))
        g = torch.sigmoid(self.gate(torch.cat([M, u], dim=-1)))    # GRU-style write gate
        if code_conf is not None:   # absolute conf-gate: a low-confidence slice writes LITTLE
            if valid_mask is not None:
                w = (code_conf * valid_mask).sum(1, keepdim=True) / valid_mask.sum(1, keepdim=True).clamp(min=1)
            else:
                w = code_conf.mean(1, keepdim=True)
            g = g * w.clamp(0.0, 1.0).unsqueeze(-1)
        M = (1.0 - g) * M + g * u                                 # CONVEX gated update -> bounded
        M = self.state_ln(M + self.ffn(self.ffn_ln(M)))          # FFN + state-norm -> ||M|| stable
        return M

    def read(self, M: torch.Tensor) -> torch.Tensor:
        return self.read_ln(M)      # [B,K,d] read-only conditioning prefix
