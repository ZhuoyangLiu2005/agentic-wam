import random

import torch
import torch.nn as nn
import torchvision.transforms as TF


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        assert x.dtype == torch.uint8
        x = x.to(torch.float32) / 255.0
        return x

class Pad(nn.Module):
    def __init__(self, padding, fill=0, padding_mode='constant'):
        super().__init__()
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
        self.pad = TF.Pad(padding=tuple(padding), fill=fill, padding_mode=padding_mode)
    
    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, "Can only pad tensor of 4 dims."
        return self.pad(x)


class ClipColorCast(nn.Module):
    """Per-clip, per-channel multiplicative colour cast.

    Models RoboTwin's `random_light` exactly: `_base_task.py` fixes a WHITE ambient term
    (`set_ambient_light([0.5, 0.5, 0.5])`) and then draws each directional/point light's RGB
    intensity as `[rand(), rand(), rand()]` ~ U[0,1]^3. Rendered radiance is therefore the albedo
    scaled INDEPENDENTLY PER CHANNEL -- a multiplicative cast, not an HSV rotation.

    This is why ColorJitter runs with `hue=0` in the RoboTwin configs: a hue rotation moves colour
    IDENTITY (red -> orange), which RoboTwin never does, and most RoboTwin training frames carry
    an instruction naming the target object's colour ("red bottle", "bottle with red cap"), so hue
    jitter would desynchronise the language<->appearance binding it is supposed to ground.

    One draw per call, so all T frames of a clip share the cast (the video-prediction target stays
    consistent with the observation); the processor calls the transform list once per camera, so
    cameras get independent draws. Input/output: [T,C,H,W] float in [0,1].
    """

    def __init__(self, low: float = 0.5, high: float = 1.0, p: float = 0.7):
        super().__init__()
        assert 0.0 < low <= high, f"need 0 < low <= high, got {low}, {high}"
        assert 0.0 <= p <= 1.0, f"p must be in [0,1], got {p}"
        self.low, self.high, self.p = float(low), float(high), float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"expected [T,C,H,W], got {tuple(x.shape)}"
        if random.random() > self.p:
            return x
        gain = torch.empty(1, x.shape[1], 1, 1, dtype=x.dtype, device=x.device).uniform_(
            self.low, self.high
        )
        return (x * gain).clamp_(0.0, 1.0)

    def extra_repr(self) -> str:
        return f"low={self.low}, high={self.high}, p={self.p}"
