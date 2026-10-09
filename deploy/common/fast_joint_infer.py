#!/usr/bin/env python
"""Compiled-step override for AGENTICWAMModel._infer_action_joint. DEPLOY-ONLY.

Why: profiling (deploy/tests/profile_joint_infer.py) showed joint tri-modal
inference is launch-overhead-bound -- GPU busy ~57% of wall, ~5k tiny kernel
launches per denoise step -- and that compiling only `model.mot` recovers just
~6% because most launches live in the per-step machinery AROUND it
(_video_pre, the three pre_dit/post_dit pairs). This module compiles the WHOLE
per-step core:

    _video_pre -> pre_dit x3 -> mot -> _video_post -> post_dit x2

into one reduce-overhead/fullgraph region (CUDA graphs), while everything
data-dependent stays eager between steps: the scheduler updates, the cond-frame
re-injection, and the LLaDA confidence unmasking (nonzero/argsort cannot live
in a CUDA graph).

Separation contract: agenticwam_model.py is NOT modified. install(model) monkey-patches
the bound method on one instance; the replacement reproduces the original's
DEFAULT decode path exactly (same op order, same RNG draws) and DEFERS TO THE
ORIGINAL whenever any research env-gate is active (AGENTICWAM_REMASK_ROUNDS, AGENTICWAM_VIZ,
AGENTICWAM_TRACE, oracle/threshold/temperature knobs, code-memory), so every
experimental path keeps its original behaviour.

All per-step-varying values enter the compiled core as tensors -- a Python
float there would insert a guard and recompile every step. A new prompt length
changes the code-canvas shape and triggers one recompile per new shape (same
behaviour the plain mot compile had). If compilation itself fails, the core
falls back to the identical eager function and logs why -- the server keeps
serving either way.
"""
import logging
import os
import types

import torch

logger = logging.getLogger(__name__)

_EXOTIC_ENV = (
    "AGENTICWAM_REMASK_ROUNDS", "AGENTICWAM_VIZ", "AGENTICWAM_TRACE", "AGENTICWAM_ORACLE_CODE_FILE",
    "AGENTICWAM_CONF_THRESH", "AGENTICWAM_PREFIX_LAST", "AGENTICWAM_DECODE_TEMP", "AGENTICWAM_ORDER_TEMP",
    "AGENTICWAM_REVEAL_CAP", "AGENTICWAM_WARMUP", "AGENTICWAM_FL_FRAC", "AGENTICWAM_MIN_CODE_LEN",
    "AGENTICWAM_ABLATE_FREEZE_VIDEO", "AGENTICWAM_ABLATE_DROP_CODE", "AGENTICWAM_ABLATE_VIT_PREFIX",
)


def _exotic_mode_active(model) -> bool:
    if any(os.environ.get(k) for k in _EXOTIC_ENV):
        return True
    if os.environ.get("AGENTICWAM_REVEAL_ORDER", "confidence") != "confidence":
        return True
    if getattr(model, "code_memory_enabled", False):
        return True
    return False


def _step_core(model, latents_video, latents_action, code_ids, sigma_v, t_a,
               prompt_embs, context, context_mask, attn_mask, cond_pos, cond_mask5d, vit_pref, prefix_ids):
    """One denoise step, noise -> predictions. Pure tensor ops; the compiled region.
    `vit_pref` is the code expert's vision prefix ([1, P, D], or None without code vision). As in the original
    path (AgenticWAMCosmos._pre_dit_vis), P placeholder ids (`prefix_ids`, built outside this region) are
    embedded ahead of the code so RoPE covers P + L positions, their embeddings are replaced by the prefix,
    and the P rows are dropped from the logits."""
    v_sig_pf = torch.where(cond_pos, model.cond_timestep, sigma_v).to(latents_video.dtype)
    video_tokens, video_aux, shape_info = model._video_pre(
        latents_video, v_sig_pf, prompt_embs, cond_mask5d)
    action_pre = model.action_expert.pre_dit(
        action_tokens=latents_action, timestep=t_a, context=context, context_mask=context_mask)
    if vit_pref is None:
        n_prefix = 0
        code_pre = model.code_expert.pre_dit(code_ids)
    else:
        n_prefix = vit_pref.shape[1]
        code_pre = model.code_expert.pre_dit(torch.cat([prefix_ids, code_ids], dim=1))
        code_tokens = code_pre["tokens"].clone()
        code_tokens[:, :n_prefix, :] = vit_pref.to(code_tokens.dtype)
        code_pre["tokens"] = code_tokens
    tokens = {"video": video_tokens, "code": code_pre["tokens"], "action": action_pre["tokens"]}
    aux = {"video": video_aux,
           "code": {"t_mod": code_pre["t_mod"], "freqs": code_pre["freqs"], "context": None},
           "action": {"t_mod": action_pre["t_mod"], "freqs": action_pre["freqs"],
                      "context": {"context": action_pre["context"],
                                  "mask": action_pre["context_mask"]}}}
    out = model.mot(tokens, aux, attn_mask)
    pred_video = model._video_post(out["video"], video_aux, shape_info)
    pred_action = model.action_expert.post_dit(out["action"], action_pre)
    code_logits = model.code_expert.post_dit(out["code"])[0][n_prefix:]   # logits in code_ids coordinates
    return pred_video, pred_action, code_logits


def install(model) -> None:
    """Replace model._infer_action_joint on THIS instance. Reversible:
    model._infer_action_joint_orig keeps the original bound method."""
    orig = model._infer_action_joint
    model._infer_action_joint_orig = orig
    state = {"core": None, "compiled": None}
    no_compile = os.environ.get("AGENTICWAM_FAST_JOINT_NO_COMPILE", "") not in ("", "0")

    def _core_dispatch(*args):
        if no_compile or state["compiled"] is False:      # disabled, or an earlier compile failed
            return _step_core(*args)
        if state["compiled"] is None:
            state["compiled"] = torch.compile(_step_core, mode="reduce-overhead", fullgraph=True)
        try:
            return state["compiled"](*args)
        except Exception as exc:  # compile/capture failure -> permanent eager fallback
            logger.warning("fast_joint_infer: compiled core failed (%s); using eager core", exc)
            state["compiled"] = False
            return _step_core(*args)

    def fast_infer_action_joint(self, obs_video, proprio, prompt_embs, action_horizon,
                                num_video_frames: int = 9, num_inference_steps: int = 20,
                                sigma_shift=None, instruction=None, max_code_tokens: int = 128,
                                return_video: bool = False):
        if _exotic_mode_active(self):
            return orig(obs_video, proprio, prompt_embs, action_horizon,
                        num_video_frames=num_video_frames,
                        num_inference_steps=num_inference_steps, sigma_shift=sigma_shift,
                        instruction=instruction, max_code_tokens=max_code_tokens,
                        return_video=return_video)

        # ---- setup: verbatim semantics of the original's default path ----
        self.dit.eval()
        device = self.device
        cdt = next(self.action_expert.parameters()).dtype
        adim = self.action_expert.action_dim
        nc = self.n_cameras
        ce = self.code_expert
        mid = ce.mask_token_id
        if proprio is not None and proprio.dim() == 2:
            proprio = proprio.unsqueeze(1)
        with torch.autocast(device_type="cuda", dtype=cdt,
                            enabled=cdt in (torch.bfloat16, torch.float16)):
            obs_latents = self._encode_video_latents(obs_video.to(device))
            vit_pref = self._vit_prefix(obs_video.to(device))     # code-vision prefix, or None
            n_prefix = int(vit_pref.shape[1]) if vit_pref is not None else 0
            prefix_ids = None
            if vit_pref is not None:
                pad = getattr(ce.tokenizer, "pad_token_id", 0) or 0
                prefix_ids = torch.full((1, n_prefix), pad, dtype=torch.long, device=device)
            Cl, Hl, Wl = obs_latents.shape[1], obs_latents.shape[3], obs_latents.shape[4]
            n_cond_lat = obs_latents.shape[2] // nc
            Tlpv = max(self._pixel_to_latent(int(num_video_frames)), n_cond_lat)
            prompt_embs = prompt_embs.to(device=device, dtype=cdt)
            if proprio is not None:
                proprio = proprio.to(device=device, dtype=cdt)
            context, context_mask = self._action_context(prompt_embs, proprio)
            if instruction is None:
                instruction = ""
            prompt = ("" if os.environ.get("AGENTICWAM_CODE_NO_INSTRUCTION", "0") == "1"
                      else f"# Instruction: {instruction}\n")
            try:
                pids = ce.tokenizer.encode(prompt, add_special_tokens=False)
            except TypeError:
                pids = ce.tokenizer.encode(prompt)
            plen = len(pids)
            gen_len = int(max_code_tokens)
            cap = int(ce.freqs.shape[0])
            if plen + gen_len > cap:
                gen_len = cap - plen
            eos = getattr(ce.tokenizer, "eos_token_id", None)
            # AGENTICWAM_CODE_PIN: freeze the episode's program. The joint path
            # regenerates the code EVERY replan; on scenes where the phase
            # belief is bistable (observed: "grasp the backpack" vs "release
            # and go home" alternating across replans in continuous mode,
            # where mid-motion captures reinforce whichever phase is
            # executing), that re-roll reverses the motion at every flip.
            # policy_runtime stages `_pinned_code_ids` (the step-0 program,
            # padded to gen_len -- SAME canvas shape, so no recompile); a
            # fully-revealed canvas means the unmasking loop naturally no-ops
            # and the code acts as fixed conditioning, like sequential mode's
            # converged belief.
            pinned = getattr(self, "_pinned_code_ids", None)
            if pinned is not None and pinned.shape[0] == gen_len:
                code_ids = torch.cat([
                    torch.tensor([pids], dtype=torch.long, device=device),
                    pinned.view(1, -1).to(device)], dim=1)
            else:
                code_ids = torch.tensor([pids + [mid] * gen_len], dtype=torch.long, device=device)

            # AGENTICWAM_NOISE_SEED: draw the SAME initial noise every request, so
            # consecutive replans differ only through the observation -- pure
            # closed-loop correction instead of ~0.05 rad of sampler lottery at
            # every chunk seam (visible as retracing on the real arm). Unset ->
            # original behaviour (fresh noise per request, global RNG).
            _seed = os.environ.get("AGENTICWAM_NOISE_SEED")
            _gen = None
            if _seed:
                _gen = torch.Generator(device=device)
                _gen.manual_seed(int(_seed))
            latents_video = torch.randn(1, Cl, nc * Tlpv, Hl, Wl, device=device, dtype=cdt,
                                        generator=_gen)
            for view in range(nc):
                d0, s0 = view * Tlpv, view * n_cond_lat
                latents_video[:, :, d0:d0 + n_cond_lat] = obs_latents[:, :, s0:s0 + n_cond_lat]
            latents_action = torch.randn(1, action_horizon, adim, device=device, dtype=cdt,
                                         generator=_gen)
            self.video_scheduler.sigmas = self._vs_sigmas0.clone().to(device)
            self.video_scheduler.timesteps = self._vs_timesteps0.clone().to(device)
            self.video_scheduler.set_timesteps(num_inference_steps, device=device)
            ts_a, deltas_a = self.action_scheduler.build_inference_schedule(
                num_inference_steps, device, cdt, shift_override=sigma_shift)
            T = len(ts_a)

            # RTC (inference-only, Physical-Intelligence style): freeze the
            # already-committed actions by INPAINTING them during denoising.
            # policy_runtime stages `_rtc_targets` ([K,64] normalized action
            # rows, chunk-start anchored) and `_rtc_weights` ([K], 1.0 = hard
            # freeze, decaying = soft mask). The scheduler is rectified flow
            # (x <- x + v*dsigma, sigma 1->0), so the velocity that lands
            # exactly on a target is v* = (x - target)/sigma; guided rows blend
            # v* with the model's prediction by weight. Training-free: the
            # model still sees a fully in-distribution denoise trajectory.
            rtc_t = getattr(self, "_rtc_targets", None)
            rtc_w = getattr(self, "_rtc_weights", None)
            if rtc_t is not None:
                rtc_t = rtc_t.to(device=device, dtype=torch.float32)[: latents_action.shape[1]]
                rtc_w = rtc_w.to(device=device, dtype=torch.float32)[: rtc_t.shape[0]]
                nts = float(self.action_scheduler.num_train_timesteps)

            # step-invariant tensors for the compiled core
            cond_pos = torch.zeros(1, nc * Tlpv, dtype=torch.bool, device=device)
            cond_mask5d = torch.zeros(1, 1, nc * Tlpv, Hl, Wl, device=device, dtype=cdt)
            for view in range(nc):
                d0 = view * Tlpv
                cond_pos[:, d0:d0 + n_cond_lat] = True
                cond_mask5d[:, :, d0:d0 + n_cond_lat] = 1.0
            # The attention mask needs the video/action token counts, which are
            # only known after _video_pre's patchify -- a one-time eager probe on
            # the first step supplies them; the mask is step-invariant after that.
            attn_mask = None

            for step_index, (t_a, d_a) in enumerate(zip(ts_a, deltas_a)):
                sigma_v = self.video_scheduler.sigmas[step_index].to(device=device, dtype=cdt)
                ts_v = self.video_scheduler.timesteps[step_index]
                ts_a_ = t_a.reshape(1).to(device=device, dtype=cdt)
                if attn_mask is None:
                    # one eager probe builds the mask with the true token counts
                    v_sig_pf = torch.where(cond_pos, self.cond_timestep, sigma_v).to(cdt)
                    video_tokens, _, _ = self._video_pre(
                        latents_video, v_sig_pf, prompt_embs, cond_mask5d)
                    a_probe = self.action_expert.pre_dit(
                        action_tokens=latents_action, timestep=ts_a_,
                        context=context, context_mask=context_mask)
                    attn_mask = self._build_mot_attention_mask(
                        video_tokens.shape[1], a_probe["tokens"].shape[1],
                        device, code_seq_len=code_ids.shape[1] + n_prefix, code_prefix_len=n_prefix)
                pred_video, pred_action, code_logits = _core_dispatch(
                    self, latents_video, latents_action, code_ids, sigma_v, ts_a_,
                    prompt_embs, context, context_mask, attn_mask, cond_pos, cond_mask5d, vit_pref, prefix_ids)

                res = self.video_scheduler.step(pred_video, ts_v, latents_video, step_index)
                latents_video = res[0] if isinstance(res, (tuple, list)) else res
                for view in range(nc):
                    d0, s0 = view * Tlpv, view * n_cond_lat
                    latents_video[:, :, d0:d0 + n_cond_lat] = obs_latents[:, :, s0:s0 + n_cond_lat]
                if rtc_t is not None:
                    sigma = max(float(t_a) / nts, 1e-4)
                    K = rtc_t.shape[0]
                    x = latents_action[:, :K].float()
                    v_star = (x - rtc_t[None]) / sigma
                    w = rtc_w[None, :, None]
                    guided = w * v_star + (1.0 - w) * pred_action[:, :K].float()
                    pred_action = pred_action.clone()
                    pred_action[:, :K] = guided.to(pred_action.dtype)
                latents_action = self.action_scheduler.step(pred_action, d_a, latents_action)

                # default LLaDA unmasking (original's else-branch with default envs)
                masked = (code_ids[0, plen:] == mid).nonzero(as_tuple=True)[0] + plen
                if masked.numel() > 0:
                    frac = min(1.0, (step_index + 1) / max(1.0, float(T)))
                    target = int(round(gen_len * frac))
                    n_reveal = max(0, target - (gen_len - int(masked.numel())))
                    if step_index == T - 1:
                        n_reveal = int(masked.numel())
                    if n_reveal > 0:
                        k = min(int(n_reveal), int(masked.numel()))
                        probs = code_logits[masked].float().softmax(-1)
                        conf, tok = probs.max(-1)
                        order = torch.argsort(conf, descending=True)[:k]
                        code_ids[0, masked[order]] = tok[order].to(code_ids.dtype)

            action_out = latents_action[0].detach().float().cpu()
            # the finished gen_len canvas, exactly re-usable as _pinned_code_ids
            self._last_code_canvas = code_ids[0, plen:].detach().clone()
            gen = [t for t in code_ids[0, plen:].tolist() if t != mid]
            if eos is not None and eos in gen:
                gen = gen[:gen.index(eos)]
            try:
                policy_code = ce.tokenizer.decode(gen, skip_special_tokens=True).strip()
            except Exception:
                policy_code = ""
            if return_video:
                gen_video = self._decode_video_latents(latents_video).detach().float().cpu()
                return action_out, policy_code, gen_video
            return action_out, policy_code

    model._infer_action_joint = types.MethodType(fast_infer_action_joint, model)
    logger.info("fast_joint_infer installed (compiled step core%s)",
                ", COMPILE DISABLED" if no_compile else "")
