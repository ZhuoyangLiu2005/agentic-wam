"""agenticwam_cosmos tri-modal MoT: Cosmos-2.5-2B video expert + Wan-style action/code experts.

The coupler (`GenericMoT`) does only the cross-expert joint self-attention; each expert
implements its own `pre_attn`/`post_attn` so the Cosmos AdaLN-Zero block and the Wan blocks
coexist unchanged. `build_agenticwam_cosmos` constructs the full model; `load_video_pretrain` loads
the video-pretrained checkpoint into it.
"""
from .generic_mot import GenericMoT
from .cosmos_video_expert import CosmosVideoExpert
from .wan_experts import WanExpertMixin, WanActionExpert, WanCodeExpert
from .agenticwam_model import AgenticWAMCosmos
from .build import build_agenticwam_cosmos, build_cosmos_dit_config, load_video_pretrain

__all__ = [
    "GenericMoT",
    "CosmosVideoExpert",
    "WanExpertMixin",
    "WanActionExpert",
    "WanCodeExpert",
    "AgenticWAMCosmos",
    "build_agenticwam_cosmos",
    "build_cosmos_dit_config",
    "load_video_pretrain",
]
