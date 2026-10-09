"""Coding Expert for the tri-modal MoT-WAM (Code | Video | Action).

The Coding Expert is a *co-equal* MoT expert: it owns a 30-layer stack of
`CodeBlock`s that participate in the per-layer joint self-attention with the
video and action experts (modality order: code -> video -> action). It generates
executable policy code autoregressively (teacher-forced CE).

Design (see plan): build the body from Qwen3-VL-2B-Instruct's 28 decoder layers
+ 2 identity-initialized layers = 30, each wrapped to the MoT block interface
(`norm1`, `self_attn.{q,k,v,o,norm_q,norm_k}`, `num_heads`, `norm2`, `ffn`,
`gate`, `modulation`). The joint-attention q/k/v are *new* Wan-style projections
(2048 -> 24*128=3072); Qwen's RMSNorms and SwiGLU MLP are reused. Code uses
*neutral* modulation (shift=0, scale=0, gate=1) because it is autoregressive, not
diffusion-noised. A tiny offline `MockCodeExpert` exposes the same interface for
CPU smoke tests.

Interface note for future joint denoising: `pre_dit`/`post_dit` return
hidden-state tokens + logits the same way regardless of how code is produced, so
a masked/diffusion code path can replace AR generation without changing the MoT
call sites.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from agenticwam.utils.logging_config import get_logger

from .wan_video_dit import GateModule, RMSNorm, SelfAttention, precompute_freqs_cis

logger = get_logger(__name__)


# Index layout of the 6 modulation chunks consumed by MoT._split_modulation:
#   0: shift_msa, 1: scale_msa, 2: gate_msa, 3: shift_mlp, 4: scale_mlp, 5: gate_mlp
# Neutral = shifts/scales 0, gates 1  -> modulate(x,0,0)=x and gate(res,1,d)=res+d.
def _neutral_modulation(hidden_dim: int, active: bool) -> torch.Tensor:
    mod = torch.zeros(1, 6, hidden_dim)
    if active:
        mod[0, 2, :] = 1.0  # gate_msa
        mod[0, 5, :] = 1.0  # gate_mlp
    # active=False -> all zeros -> gate_msa=gate_mlp=0 -> block is exact identity
    return mod


def _interp_linear_(dst: nn.Linear, src_weight: torch.Tensor, src_bias: Optional[torch.Tensor]) -> None:
    """Best-effort init of a Linear's weight/bias from a differently-shaped source
    via 2D interpolation (ActionDiT precedent). Falls back silently to default init."""
    try:
        with torch.no_grad():
            w = src_weight.detach().float()  # [out_src, in_src]
            out_dst, in_dst = dst.weight.shape
            w = w.view(1, 1, w.shape[0], w.shape[1])
            w = F.interpolate(w, size=(out_dst, in_dst), mode="bilinear", align_corners=False)
            dst.weight.copy_(w.view(out_dst, in_dst).to(dst.weight.dtype))
            if dst.bias is not None and src_bias is not None:
                b = src_bias.detach().float().view(1, 1, -1)
                b = F.interpolate(b, size=(out_dst,), mode="linear", align_corners=False)
                dst.bias.copy_(b.view(out_dst).to(dst.bias.dtype))
    except Exception as exc:  # pragma: no cover - init is best-effort
        logger.warning("CodeBlock attn projection interp-init failed (%s); using default init.", exc)


class CodeBlock(nn.Module):
    """One code-expert layer, exposing the MoT block interface.

    norm1/norm2/ffn come from a Qwen decoder layer (pretrained) when provided;
    the joint-attention projections (`self_attn`) are new Wan-style projections so
    q/k/v live in the shared 24*128=3072 attention space.
    """

    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        active: bool = True,
        norm1: Optional[nn.Module] = None,
        norm2: Optional[nn.Module] = None,
        ffn: Optional[nn.Module] = None,
        ffn_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        # `active=False` blocks are the identity-initialized EXTRA layers (gate=0);
        # they are NEW capacity and stay trainable even when the Qwen body is frozen.
        self.is_extra = not active

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        # norm1/norm2 feed attention / FFN. Reuse Qwen RMSNorms when given so the
        # pretrained FFN sees inputs with the statistics it expects.
        self.norm1 = norm1 if norm1 is not None else RMSNorm(hidden_dim, eps=eps)
        self.norm2 = norm2 if norm2 is not None else RMSNorm(hidden_dim, eps=eps)
        if ffn is not None:
            self.ffn = ffn
        else:
            inner = ffn_dim if ffn_dim is not None else hidden_dim * 4
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, inner), nn.GELU(approximate="tanh"), nn.Linear(inner, hidden_dim)
            )
        self.gate = GateModule()
        # Learnable modulation, initialized neutral (active) or identity (extra layers).
        self.modulation = nn.Parameter(_neutral_modulation(hidden_dim, active=active))
        # code uses no cross-attention; context_payload is always None for the code expert.
        self.cross_attn = None
        self.norm3 = None


class CodeExpert(nn.Module):
    """Qwen-bodied 30-layer code expert (a co-equal MoT expert).

    Exposes `.blocks`, `.num_heads`, `.attn_head_dim`, `.use_gradient_checkpointing`
    for the MoT, plus `pre_dit`/`post_dit`/`code_loss`/`generate` for AR code.
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        embed_tokens: nn.Module,
        final_norm: nn.Module,
        lm_head: nn.Module,
        hidden_dim: int,
        num_heads: int,
        attn_head_dim: int,
        vocab_size: int,
        max_code_tokens: int = 512,
        rope_theta: float = 10000.0,
        use_gradient_checkpointing: bool = False,
        tokenizer: Optional[Any] = None,
        freeze_backbone: bool = False,
        code_objective: str = "ar",
        code_block_length: int = 4,
        mask_token_id: Optional[int] = None,
        code_canvas: int = 0,     # >0: FIXED code canvas (EOS-filled, fully supervised); 0 = batch-max (legacy)
    ):
        super().__init__()
        self.blocks = blocks
        self.embed_tokens = embed_tokens
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.hidden_dim = hidden_dim
        # masked-diffusion (LLaDA-style): substitute this learnable embedding at masked
        # code positions; lm_head still predicts over the original vocab.
        # non-trainable constant mask embedding (avoids a conditionally-used trainable
        # param desyncing ZeRO grad collectives when a step masks 0 tokens at batch_size=1).
        self.register_buffer("mask_embed", torch.zeros(hidden_dim), persistent=True)
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.vocab_size = vocab_size
        self.max_code_tokens = int(max_code_tokens)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.tokenizer = tokenizer
        self.freeze_backbone = bool(freeze_backbone)
        # block-diffusion (SDAR) code objective: 'ar' (next-token) | 'blockdiff' (absorbing)
        self.code_objective = str(code_objective)
        self.code_block_length = int(code_block_length)
        self.mask_token_id = mask_token_id
        # FIXED CODE CANVAS (0 = off -> legacy batch-max alignment, byte-identical). When >0 the code
        # region of every sample is exactly this many positions: program + EOS, EOS-filled to the
        # canvas, with labels on ALL of them. Matches the inference geometry (an all-<MASK> canvas of
        # AGENTICWAM_INFER_CODE_TOKENS positions), which batch-max alignment never showed the model.
        self.code_canvas = int(code_canvas or 0)
        # RoPE cache for the code positions (1D causal sequence).
        self.register_buffer(
            "freqs",
            precompute_freqs_cis(attn_head_dim, end=max(self.max_code_tokens + 8, 1024), theta=rope_theta),
            persistent=False,
        )
        # J4: timestep-conditioned AdaLN for continuous-embedding code diffusion (Motus-style).
        # Zero-init the last layer so compute_t_mod returns 0 at init == the prior neutral t_mod
        # (so an old checkpoint loaded strict=False behaves identically until trained).
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        nn.init.zeros_(self.time_embed[-1].weight)
        nn.init.zeros_(self.time_embed[-1].bias)

    # --- MoT-facing methods --------------------------------------------------
    def compute_t_mod(self, timestep: torch.Tensor) -> torch.Tensor:
        """J4: map a per-sample diffusion timestep in [0,1] to AdaLN modulation [B, 6, hidden_dim]
        (added to the learnable per-block `modulation` base inside MoT). Sinusoidal embed -> MLP."""
        d = self.hidden_dim
        half = d // 2
        device = timestep.device
        idx = torch.arange(half, device=device, dtype=torch.float32)
        freqs = torch.exp(idx * (-9.21034037 / max(half, 1)))  # -ln(10000)/half
        args = timestep.float().reshape(-1, 1) * 1000.0 * freqs.reshape(1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, 2*half]
        if emb.shape[-1] < d:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], d - emb.shape[-1], device=device)], dim=-1)
        w0 = self.time_embed[0].weight
        out = self.time_embed(emb.to(dtype=w0.dtype))  # [B, 6*d]
        return out.view(-1, 6, d)

    def pre_dit(self, input_ids: torch.Tensor, **_: Any) -> Dict[str, Any]:
        """input_ids: [B, S] long. Returns tokens/freqs/t_mod (neutral)/context(None)."""
        if input_ids.ndim != 2:
            raise ValueError(f"`input_ids` must be 2D [B, S], got {tuple(input_ids.shape)}")
        b, s = input_ids.shape
        if s > self.freqs.shape[0]:
            raise ValueError(f"code seq len {s} exceeds RoPE cache {self.freqs.shape[0]}")
        tokens = self.embed_tokens(input_ids)  # [B, S, D]
        dev, dt = tokens.device, tokens.dtype
        freqs = self.freqs[:s].view(s, 1, -1).to(dev)
        t_mod = torch.zeros(b, 6, self.hidden_dim, device=dev, dtype=dt)  # neutral; modulation buffer carries gates
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t_mod": t_mod,
            "context": None,
            "context_mask": None,
            "meta": {"batch_size": b, "seq_len": s},
        }

    def post_dit(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, S, D] -> logits [B, S, vocab]."""
        return self.lm_head(self.final_norm(tokens))

    def code_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """AR: causal-LM CE with shift. blockdiff: absorbing-diffusion CE at each token's OWN
        position (no shift), over masked code positions (label!=-100). labels==-100 ignored."""
        if getattr(self, "code_objective", "ar") in ("blockdiff", "joint"):
            flat_logits = logits.reshape(-1, logits.size(-1)).float()
            flat_labels = labels.reshape(-1)
        else:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1)).float()
            flat_labels = shift_labels.view(-1)
        # reduction="sum"/valid-count (not "mean") so an all-masked sample (code-coverage
        # ablation: an instruction with no GT code) yields a finite 0 loss instead of NaN,
        # while keeping the CE op in the graph on EVERY rank. This lets the caller compute
        # code_loss unconditionally -> identical autograd structure across ranks -> DDP/ZeRO
        # gradient collectives stay in sync regardless of per-rank code coverage.
        import os as _os
        _ew = float(_os.environ.get("AGENTICWAM_PAD_EOS_WEIGHT", "1.0") or 1.0)
        _eid = getattr(self.tokenizer, "eos_token_id", None)
        if _ew != 1.0 and _eid is not None:
            # EOS-downweighted CE (weight is an env constant -> identical autograd structure
            # on every rank, so DDP/ZeRO collectives stay in sync). Weighted-mean denominator.
            _ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="none")
            _w = torch.ones_like(_ce)
            _w[flat_labels == _eid] = _ew
            _w[flat_labels == -100] = 0.0
            loss = (_ce * _w).sum() / _w.sum().clamp(min=1e-6)
        else:
            loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="sum") / (flat_labels != -100).sum().clamp(min=1)
        with torch.no_grad():
            valid = flat_labels != -100
            if valid.any():
                pred = flat_logits.argmax(dim=-1)
                acc = (pred[valid] == flat_labels[valid]).float().mean()
            else:
                acc = torch.zeros((), device=logits.device)
        out = {"loss": loss, "acc": acc}
        if _eid is not None and (_ew != 1.0 or int(_os.environ.get("AGENTICWAM_PAD_EOS_WINDOW", "0") or 0) > 0
                                 or int(getattr(self, "code_canvas", 0) or 0) > 0):
            # comparability metric: acc over non-EOS labels (EOS-window tokens are easy and
            # would inflate code_acc vs the historical curves)
            with torch.no_grad():
                _nv = valid & (flat_labels != _eid)
                out["acc_noneos"] = ((pred[_nv] == flat_labels[_nv]).float().mean()
                                     if bool(_nv.any()) else torch.zeros((), device=logits.device))
        return out

    def set_backbone_frozen(self, frozen: bool = True) -> None:
        """Freeze/unfreeze the pretrained Qwen body. Trainable regardless:
        the NEW joint-attention projections (`self_attn`), learnable `modulation`,
        and the EXTRA (`is_extra`) layers' norm/ffn. Frozen when `frozen`:
        base-layer norm1/norm2/ffn + embeddings + lm_head + final_norm."""
        self.freeze_backbone = bool(frozen)
        req = not frozen
        for blk in self.blocks:
            blk.self_attn.requires_grad_(True)       # new projections always train
            blk.modulation.requires_grad_(True)      # learnable modulation always trains
            body_req = True if blk.is_extra else req
            for m in (blk.norm1, blk.norm2, blk.ffn):
                m.requires_grad_(body_req)
        for m in (self.embed_tokens, self.lm_head, self.final_norm):
            m.requires_grad_(req)

    def build_code_batch(self, instructions, codes, device, max_length: Optional[int] = None):
        """Tokenize (instruction prompt + policy code) into input_ids/labels for
        teacher-forced CE. Prompt tokens and padding are masked to -100. Works with
        a HF tokenizer or the offline ByteTokenizer."""
        max_length = int(max_length or self.max_code_tokens)
        if isinstance(instructions, str):
            instructions = [instructions]
        if isinstance(codes, str):
            codes = [codes]
        if codes is None:
            raise ValueError("`codes` (policy_code) required to build the code batch.")
        bsz = len(codes)
        instructions = instructions if instructions is not None else [""] * bsz
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0

        def _enc(text):
            try:
                return list(self.tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                return list(self.tokenizer.encode(text))

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        # AGENTICWAM_CODE_NO_INSTRUCTION=1 -> NO text prefix: the code stream must read the scene
        # through its video attention (text-shortcut removal; see high-mask negative result).
        import os as _os
        _noinstr = _os.environ.get("AGENTICWAM_CODE_NO_INSTRUCTION", "0") == "1"
        if _noinstr and not getattr(self, "_noinstr_logged", False):
            self._noinstr_logged = True
            print("[code-no-instruction] ACTIVE: code expert receives NO text prompt", flush=True)
        # PAD-EOS WINDOW (env-gated, default 0 = byte-identical): supervise the first W pad
        # positions after the program as EOS (bounded LLaDA-style EOS padding). Teaches the
        # "program end -> EOS run" boundary for length control WITHOUT full-tail EOS padding,
        # under which EOS becomes the majority label and confidence-ordered decode floods the
        # canvas with high-confidence EOS before content is committed.
        _pew = int(_os.environ.get("AGENTICWAM_PAD_EOS_WINDOW", "0") or 0)
        if _pew > 0 and not getattr(self, "_pad_eos_logged", False):
            self._pad_eos_logged = True
            print(f"[code-pad-eos] ACTIVE window={_pew} weight={_os.environ.get('AGENTICWAM_PAD_EOS_WEIGHT', '1.0')}", flush=True)
        # FIXED CANVAS: config `code_canvas`, overridable by AGENTICWAM_CODE_CANVAS for ablations.
        _cv = int(_os.environ.get("AGENTICWAM_CODE_CANVAS", getattr(self, "code_canvas", 0)) or 0)
        if _cv > 0 and not getattr(self, "_canvas_logged", False):
            self._canvas_logged = True
            print(f"[code-canvas] ACTIVE fixed canvas={_cv}, EOS-filled, labels on the whole canvas", flush=True)
        seqs, labs = [], []
        for i in range(bsz):
            if _noinstr:
                p_ids = []
            else:
                # [dataprep] datasets with code slices pass the fully rendered prompt (instruction +
                # helper library, already starting with "# Instruction:"); other paths pass the bare instruction.
                ptxt = str(instructions[i])
                if not ptxt.startswith("# Instruction:"):
                    ptxt = f"# Instruction: {ptxt}\n"
                p_ids = _enc(ptxt)
            c_ids = _enc(codes[i])
            if _cv > 0 and c_ids:
                # program + terminal EOS, then EOS-filled to exactly _cv. Truncate to _cv-1 first so
                # the terminal EOS survives even if a program ever outgrows the canvas.
                if eos_id is not None:
                    c_ids = c_ids[:_cv - 1] + [eos_id]
                    if len(c_ids) < _cv:
                        c_ids = c_ids + [eos_id] * (_cv - len(c_ids))
                else:
                    c_ids = c_ids[:_cv]
            elif c_ids and eos_id is not None:
                c_ids = c_ids + [eos_id]  # teach the model to STOP after the code (else greedy decoding runs to the token cap and degenerates)
            # else: empty code (coverage ablation / no GT) -> no labels -> fully masked below (no code loss for this sample)
            if _cv > 0 and len(p_ids) + len(c_ids) > max_length and not getattr(self, "_canvas_trunc_logged", False):
                self._canvas_trunc_logged = True
                print(f"[code-canvas] WARNING prompt {len(p_ids)} + canvas {_cv} > max_code_tokens "
                      f"{max_length}: the canvas is being CUT. Raise max_code_tokens.", flush=True)
            ids = (p_ids + c_ids)[:max_length]
            lab = ([-100] * len(p_ids) + c_ids)[:max_length]
            if _cv == 0 and _pew > 0 and c_ids and eos_id is not None and len(ids) < max_length:
                _n = min(_pew, max_length - len(ids))
                ids = ids + [eos_id] * _n
                lab = lab + [eos_id] * _n
            seqs.append(ids)
            labs.append(lab)
        m = max(len(s) for s in seqs) if seqs else 1
        m = min(m, max_length)
        input_ids, labels = [], []
        for ids, lab in zip(seqs, labs):
            pad = m - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(lab + [-100] * pad)
        ii = torch.tensor(input_ids, dtype=torch.long, device=device)
        lab = torch.tensor(labels, dtype=torch.long, device=device)
        if getattr(self, "code_objective", "ar") in ("blockdiff", "joint"):
            ii, lab = self._absorb_mask(ii, lab)
        return ii, lab

    def _absorb_mask(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """Absorbing noising for the diffusion code expert. code_objective:
          'joint'     -> LLaDA GLOBAL masking: one mask-ratio t~U(0,1) per sample over ALL code
                         tokens (the code's single 'timestep', to co-denoise with video/action under
                         fully-bidirectional code attention).
          'blockdiff' -> SDAR per-block masking: independent t_block~U(0,1) per block-length block
                         (block-causal attention).
        Loss keeps labels ONLY at masked positions; prompt/pad stay -100."""
        mid = self.mask_token_id
        if mid is None:
            return input_ids, labels
        inp = input_ids.clone(); lb = labels.clone()
        B, L = inp.shape
        dev = inp.device
        joint = (getattr(self, "code_objective", "") == "joint")
        bl = max(1, int(getattr(self, "code_block_length", 4)))
        nblk = (L + bl - 1) // bl
        # HIGH-MASK CURRICULUM (env-gated; defaults reproduce the historical t~U(0,1) exactly,
        # including the RNG stream). WHY: the conditioning probe (2026-08-11, step-21k ckpt)
        # showed the code stream is LANGUAGE-dominant -- swapped-instruction generations match
        # the instruction's program at 95% of the matched baseline while overlap with the true
        # scene's program sits BELOW the GT-GT boilerplate floor. With t~U(0,1) half the
        # gradient is spent infilling tokens the visible code context makes trivial; inference
        # is ALL-mask. Raising t_min / forcing t=1.0 concentrates gradient on the inference
        # regime and on scene-specific tokens, and also shrinks the action->code attention gap
        # (action trains against mostly-masked code, as it sees at inference).
        #   AGENTICWAM_CODE_MASK_TMIN  in [0,1): t ~ U(t_min, 1.0)
        #   AGENTICWAM_CODE_MASK_FULLP in [0,1]: with this prob, t = 1.0 exactly
        import os as _os
        _tmin = float(_os.environ.get("AGENTICWAM_CODE_MASK_TMIN", "0.0"))
        _fullp = float(_os.environ.get("AGENTICWAM_CODE_MASK_FULLP", "0.0"))
        if not getattr(self, "_mask_curriculum_logged", False):
            self._mask_curriculum_logged = True
            if _tmin > 0.0 or _fullp > 0.0:
                print(f"[code-mask-curriculum] ACTIVE t_min={_tmin} full_p={_fullp}", flush=True)
        for b in range(B):
            is_code = labels[b] != -100
            if joint:
                if _fullp > 0.0 and float(torch.rand(1)) < _fullp:
                    t = 1.0
                else:
                    t = float(torch.empty(1, device=dev).uniform_(_tmin, 1.0))
                tok_t = torch.full((L,), t, device=dev)
            else:
                tok_t = torch.rand(nblk, device=dev).repeat_interleave(bl)[:L]
            mask_here = is_code & (torch.rand(L, device=dev) < tok_t)
            if not bool(mask_here.any()):
                cp = is_code.nonzero(as_tuple=True)[0]
                if cp.numel() > 0:
                    mask_here[int(cp[0])] = True
            inp[b, mask_here] = mid
            lb[b, is_code & (~mask_here)] = -100
        return inp, lb

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256, eos_token_id: Optional[int] = None) -> torch.Tensor:
        """Standalone greedy AR generation (code-only self-attention; no MoT joint attention).
        Used for qualitative eval / the generate script."""
        self.eval()
        ids = input_ids
        for _ in range(max_new_tokens):
            s = ids.shape[1]
            if s > self.freqs.shape[0]:
                break
            logits = self._forward_code_only(ids)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
        return ids

    def _forward_code_only(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the code stack with code-only causal self-attention (no video/action).
        Mirrors the MoT per-block math so generation matches training-time blocks."""
        from .wan_video_dit import flash_attention, modulate, rope_apply

        pre = self.pre_dit(input_ids)
        x, freqs, t_mod = pre["tokens"], pre["freqs"], pre["t_mod"]
        s = x.shape[1]
        causal = torch.tril(torch.ones(s, s, dtype=torch.bool, device=x.device))
        for block in self.blocks:
            base = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
            sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = (base + t_mod).chunk(6, dim=1)
            attn_in = modulate(block.norm1(x), sh_msa, sc_msa)
            q = rope_apply(block.self_attn.norm_q(block.self_attn.q(attn_in)), freqs, block.num_heads)
            k = rope_apply(block.self_attn.norm_k(block.self_attn.k(attn_in)), freqs, block.num_heads)
            v = block.self_attn.v(attn_in)
            attn = flash_attention(q=q, k=k, v=v, num_heads=block.num_heads, ctx_mask=causal)
            x = block.gate(x, g_msa, block.self_attn.o(attn))
            mlp_in = modulate(block.norm2(x), sh_mlp, sc_mlp)
            x = block.gate(x, g_mlp, block.ffn(mlp_in))
        return self.post_dit(x)


# --------------------------------------------------------------------------- #
# Mock backend (offline, tiny) for CPU smoke tests / unit tests.
# --------------------------------------------------------------------------- #
class ByteTokenizer:
    """Trivial offline tokenizer over UTF-8 bytes (vocab 256 + pad/eos)."""

    pad_token_id = 256
    eos_token_id = 257
    vocab_size = 258

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def __call__(self, texts, max_length=512, return_tensors=None, padding=True, truncation=True):
        if isinstance(texts, str):
            texts = [texts]
        seqs = [self.encode(t)[: max_length - 1] + [self.eos_token_id] for t in texts]
        m = max(len(s) for s in seqs)
        ids = [s + [self.pad_token_id] * (m - len(s)) for s in seqs]
        mask = [[1] * len(s) + [0] * (m - len(s)) for s in seqs]
        ids_t = torch.tensor(ids, dtype=torch.long)
        mask_t = torch.tensor(mask, dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": ids_t, "attention_mask": mask_t}
        return {"input_ids": ids_t, "attention_mask": mask_t}


def _build_mock_code_expert(
    num_layers: int,
    num_heads: int,
    attn_head_dim: int,
    hidden_dim: int,
    vocab_size: int = ByteTokenizer.vocab_size,
    max_code_tokens: int = 512,
    use_gradient_checkpointing: bool = False,
) -> CodeExpert:
    blocks = nn.ModuleList(
        [
            CodeBlock(
                hidden_dim=hidden_dim,
                attn_head_dim=attn_head_dim,
                num_heads=num_heads,
                active=(i < num_layers),  # all active for mock
                ffn_dim=hidden_dim * 2,
            )
            for i in range(num_layers)
        ]
    )
    embed = nn.Embedding(vocab_size, hidden_dim)
    final_norm = RMSNorm(hidden_dim)
    lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
    expert = CodeExpert(
        blocks=blocks,
        embed_tokens=embed,
        final_norm=final_norm,
        lm_head=lm_head,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        attn_head_dim=attn_head_dim,
        vocab_size=vocab_size,
        max_code_tokens=max_code_tokens,
        use_gradient_checkpointing=use_gradient_checkpointing,
        tokenizer=ByteTokenizer(),
    )
    return expert


# --------------------------------------------------------------------------- #
# Qwen3-VL backend.
# --------------------------------------------------------------------------- #
def _build_qwen_code_expert(
    model_name: str,
    num_layers: int,
    num_heads: int,
    attn_head_dim: int,
    n_extra_layers: int,
    max_code_tokens: int,
    freeze_backbone: bool,
    use_lora: bool,
    device: str,
    torch_dtype: torch.dtype,
    interp_init_attn: bool = True,
) -> CodeExpert:
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    logger.info("Loading code-expert backbone %s ...", model_name)
    vlm = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch_dtype)
    text_model = vlm.model.language_model  # decoder: .layers, .norm, .embed_tokens
    qwen_layers = text_model.layers
    hidden_dim = qwen_layers[0].self_attn.q_proj.in_features  # 2048
    eps = float(getattr(getattr(vlm.config, "text_config", vlm.config), "rms_norm_eps", 1e-6))
    rope_theta = float(getattr(getattr(vlm.config, "text_config", vlm.config), "rope_theta", 10000.0))
    n_base = len(qwen_layers)
    if num_layers != n_base + n_extra_layers:
        logger.warning(
            "num_layers(%d) != qwen_layers(%d)+n_extra(%d); using qwen+extra=%d.",
            num_layers, n_base, n_extra_layers, n_base + n_extra_layers,
        )
        num_layers = n_base + n_extra_layers

    blocks = nn.ModuleList()
    for i in range(num_layers):
        src = qwen_layers[i] if i < n_base else qwen_layers[n_base - 1]  # duplicate last for extras
        active = i < n_base  # extra layers start as identity (gate=0)
        blk = CodeBlock(
            hidden_dim=hidden_dim,
            attn_head_dim=attn_head_dim,
            num_heads=num_heads,
            eps=eps,
            active=active,
            norm1=src.input_layernorm,
            norm2=src.post_attention_layernorm,
            ffn=src.mlp,
        )
        if interp_init_attn:
            sa = src.self_attn
            _interp_linear_(blk.self_attn.q, sa.q_proj.weight, getattr(sa.q_proj, "bias", None))
            _interp_linear_(blk.self_attn.k, sa.k_proj.weight, getattr(sa.k_proj, "bias", None))
            _interp_linear_(blk.self_attn.v, sa.v_proj.weight, getattr(sa.v_proj, "bias", None))
            _interp_linear_(blk.self_attn.o, sa.o_proj.weight, getattr(sa.o_proj, "bias", None))
        blocks.append(blk)

    embed = text_model.embed_tokens
    final_norm = text_model.norm
    lm_head = vlm.lm_head
    vocab_size = embed.num_embeddings

    try:
        tokenizer = AutoProcessor.from_pretrained(model_name).tokenizer
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    expert = CodeExpert(
        blocks=blocks,
        embed_tokens=embed,
        final_norm=final_norm,
        lm_head=lm_head,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        attn_head_dim=attn_head_dim,
        vocab_size=vocab_size,
        max_code_tokens=max_code_tokens,
        rope_theta=rope_theta,
        tokenizer=tokenizer,
        freeze_backbone=freeze_backbone,
    )

    # Freeze the pretrained Qwen body (base-layer norms + FFN + embeddings + lm_head);
    # the NEW joint-attn projections, learnable modulation, and extra layers stay trainable.
    expert.set_backbone_frozen(freeze_backbone)
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model  # noqa: F401
            logger.info("use_lora=True: apply LoRA to the Qwen FFN externally (hook left for follow-up).")
        except Exception as exc:
            logger.warning("peft unavailable (%s); proceeding without LoRA.", exc)

    # Free the vision tower we don't use in v1.
    del vlm
    return expert.to(device=device, dtype=torch_dtype)


def _build_sdar_code_expert(
    model_name: str,
    num_layers: int = 28,
    num_heads: int = 16,
    attn_head_dim: int = 128,
    n_extra_layers: int = 0,
    max_code_tokens: int = 512,
    code_canvas: int = 0,
    freeze_backbone: bool = False,
    interp_init_attn: bool = True,
    rope_theta: float = 1000000.0,
    eps: float = 1e-6,
    block_length: int = 4,
    objective: str = "blockdiff",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> CodeExpert:
    """Block-DIFFUSION code expert from an SDAR-1.7B checkpoint (Qwen3-1.7B body, block-diffusion
    pretrained). Architecturally identical to the Qwen3-VL LM slot (28L/2048/16H/128d/8KV/6144),
    so map model.layers -> MoT CodeBlocks exactly as the Qwen builder, keep SDAR's embed/lm_head/
    norm + its native <|MASK|> token, and set code_objective='blockdiff'. Vision reaches the code
    via MoT joint attention (code->video), so no vision encoder is needed here."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info("Loading SDAR code-expert backbone %s ...", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch_dtype, attn_implementation="sdpa")
    text_model = model.model
    qwen_layers = text_model.layers
    hidden_dim = qwen_layers[0].self_attn.q_proj.in_features
    n_base = len(qwen_layers)
    if num_layers != n_base + n_extra_layers:
        num_layers = n_base + n_extra_layers
    blocks = nn.ModuleList()
    for i in range(num_layers):
        src = qwen_layers[i] if i < n_base else qwen_layers[n_base - 1]
        active = i < n_base
        blk = CodeBlock(
            hidden_dim=hidden_dim, attn_head_dim=attn_head_dim, num_heads=num_heads, eps=eps,
            active=active, norm1=src.input_layernorm, norm2=src.post_attention_layernorm, ffn=src.mlp)
        if interp_init_attn:
            sa = src.self_attn
            _interp_linear_(blk.self_attn.q, sa.q_proj.weight, getattr(sa.q_proj, "bias", None))
            _interp_linear_(blk.self_attn.k, sa.k_proj.weight, getattr(sa.k_proj, "bias", None))
            _interp_linear_(blk.self_attn.v, sa.v_proj.weight, getattr(sa.v_proj, "bias", None))
            _interp_linear_(blk.self_attn.o, sa.o_proj.weight, getattr(sa.o_proj, "bias", None))
        blocks.append(blk)
    embed = text_model.embed_tokens
    final_norm = text_model.norm
    lm_head = model.lm_head
    vocab_size = embed.num_embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mask_id = getattr(tokenizer, "mask_token_id", None)
    expert = CodeExpert(
        blocks=blocks, embed_tokens=embed, final_norm=final_norm, lm_head=lm_head,
        hidden_dim=hidden_dim, num_heads=num_heads, attn_head_dim=attn_head_dim,
        vocab_size=vocab_size, max_code_tokens=max_code_tokens, rope_theta=rope_theta,
        tokenizer=tokenizer, freeze_backbone=freeze_backbone,
        code_objective=objective, code_block_length=block_length, mask_token_id=mask_id,
        code_canvas=code_canvas)
    expert.set_backbone_frozen(freeze_backbone)
    del model
    logger.info("SDAR code expert: %d layers, hidden=%d, vocab=%d, mask_id=%s",
                num_layers, hidden_dim, vocab_size, mask_id)
    return expert.to(device=device, dtype=torch_dtype)


def build_code_expert(
    model_name: str = "mock",
    num_layers: int = 30,
    num_heads: int = 24,
    attn_head_dim: int = 128,
    n_extra_layers: int = 2,
    hidden_dim: int = 2048,
    max_code_tokens: int = 512,
    code_canvas: int = 0,
    freeze_backbone: bool = True,
    use_lora: bool = False,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    mock_hidden_dim: int = 256,
    objective: str = "ar",
    block_length: int = 4,
) -> CodeExpert:
    """Factory. `model_name="mock"` -> tiny offline expert; else a Qwen3-VL-bodied expert.

    For the mock path, `num_heads/attn_head_dim/num_layers` must match the other
    MoT experts used in the (tiny) smoke test.
    """
    if objective in ("blockdiff", "joint") or "sdar" in str(model_name).lower():
        _obj = objective if objective in ("blockdiff", "joint", "ar") else "blockdiff"  # respect explicit mode (ar runs on SDAR too)
        return _build_sdar_code_expert(
            model_name=model_name, num_layers=num_layers, num_heads=num_heads,
            attn_head_dim=attn_head_dim, n_extra_layers=n_extra_layers,
            max_code_tokens=max_code_tokens, code_canvas=code_canvas, freeze_backbone=freeze_backbone,
            block_length=block_length, objective=_obj, device=device, torch_dtype=torch_dtype,
        )
    if model_name == "mock":
        return _build_mock_code_expert(
            num_layers=num_layers,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            hidden_dim=mock_hidden_dim,
            max_code_tokens=max_code_tokens,
        ).to(device=device, dtype=torch_dtype)
    return _build_qwen_code_expert(
        model_name=model_name,
        num_layers=num_layers,
        num_heads=num_heads,
        attn_head_dim=attn_head_dim,
        n_extra_layers=n_extra_layers,
        max_code_tokens=max_code_tokens,
        freeze_backbone=freeze_backbone,
        use_lora=use_lora,
        device=device,
        torch_dtype=torch_dtype,
    )
