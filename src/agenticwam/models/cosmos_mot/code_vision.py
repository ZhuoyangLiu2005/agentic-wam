"""Semantic vision prefix for the code expert (planner).

Frozen Qwen3-VL-2B vision tower (24L/1024h, patch16, 2x2 merge) whose merger
outputs 2048 == SDAR hidden, so the merged tokens drop straight into the code
stream as a conditioning prefix: [ViT tokens][instruction][code canvas].
The tower is frozen; only the merger (the natural projector, pretrained to
align with a Qwen3-family embedding space) trains. Input obs frames are the
same pixels the VAE consumes ([-1,1]); at 224x320 each view yields 70 merged
tokens (14x20 patches / 2x2), so 3 views -> a 210-token prefix.
"""
import json
import os

import torch
import torch.nn as nn


class CodeVisionPrefix(nn.Module):
    def __init__(self, snapshot_path: str, dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        cfg = AutoConfig.from_pretrained(snapshot_path)
        self.vcfg = cfg.vision_config
        self.vit = Qwen3VLVisionModel(self.vcfg)
        # selective load of the visual.* weights from the single-file safetensors
        from safetensors.torch import load_file
        sf = os.path.join(snapshot_path, "model.safetensors")
        sd = load_file(sf)
        vsd = {k.split("visual.", 1)[1]: v for k, v in sd.items() if "visual." in k}
        missing, unexpected = self.vit.load_state_dict(vsd, strict=False)
        print(f"[code_vision] loaded visual.*: {len(vsd)} keys "
              f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)
        self.vit.to(dtype)
        self.vit.requires_grad_(False)
        # train only the merger/projector
        merger = getattr(self.vit, "merger", None)
        if merger is not None:
            merger.requires_grad_(True)
        pre = json.load(open(os.path.join(snapshot_path, "preprocessor_config.json")))
        self.register_buffer("img_mean", torch.tensor(pre["image_mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor(pre["image_std"]).view(1, 3, 1, 1), persistent=False)
        self.patch = int(self.vcfg.patch_size)              # 16
        self.tpatch = int(getattr(self.vcfg, "temporal_patch_size", 2))
        self.merge = int(self.vcfg.spatial_merge_size)      # 2

    @property
    def out_dim(self):
        return int(getattr(self.vcfg, "out_hidden_size", 2048))

    def tokens_per_view(self, H, W):
        return (H // self.patch) * (W // self.patch) // (self.merge ** 2)

    def forward(self, obs_frames: torch.Tensor) -> torch.Tensor:
        """obs_frames: [B, nc, 3, H, W] in [-1, 1] (same range the VAE consumes).
        returns [B, nc * tokens_per_view, out_dim]."""
        B, nc, C, H, W = obs_frames.shape
        p, tp, m = self.patch, self.tpatch, self.merge
        assert H % (p * m) == 0 and W % (p * m) == 0, f"H/W must be multiples of {p*m}"
        dev = self.img_mean.device
        x = obs_frames.reshape(B * nc, C, H, W).to(device=dev, dtype=torch.float32)
        x = (x + 1.0) / 2.0
        x = (x - self.img_mean) / self.img_std
        x = x.unsqueeze(2).expand(-1, -1, tp, -1, -1)        # temporal patch: repeat frame
        gh, gw = H // p, W // p
        # Qwen-VL patch flatten (matches Qwen2VLImageProcessor): merge-window raster order
        x = x.reshape(B * nc, C, tp, gh // m, m, p, gw // m, m, p)
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)             # [BN, gh/m, gw/m, m, m, tp, C, p, p]
        x = x.reshape(B * nc * gh * gw, C * tp * p * p)
        grid = torch.tensor([[1, gh, gw]] * (B * nc), device=dev, dtype=torch.long)
        vdt = next(self.vit.parameters()).dtype
        out = self.vit(x.to(vdt), grid)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        tpv = self.tokens_per_view(H, W)
        return out.reshape(B, nc * tpv, self.out_dim)
