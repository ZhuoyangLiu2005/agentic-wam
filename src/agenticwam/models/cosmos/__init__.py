"""Cosmos-Predict2.5 components: the DiT, the video tokenizer, the Wan2.2 VAE, the flow-matching schedulers.

Heavy modules (the Cosmos-Reason1 text encoder) are NOT imported at package load — access them via their
submodules directly to avoid pulling transformers when only the DiT/VAE/tokenizer are needed.
"""
from .transformer_cosmos import (
    COSMOS_2B_CONFIG,
    COSMOS_14B_CONFIG,
    CosmosTransformer3DModel,
    CosmosTransformerConfig,
    CosmosRotaryPosEmbed,
    MultiViewRoPE,
    apply_rotary_emb,
)
from .video_tokenizer import VideoTokenizer
from .scheduler import (
    BaseFlowMatchScheduler,
    BetaFlowMatchEulerScheduler,
    BetaSchedulerConfig,
    FlowMatchUniPCScheduler,
    SchedulerConfig,
)

__all__ = [
    "COSMOS_2B_CONFIG",
    "COSMOS_14B_CONFIG",
    "CosmosTransformer3DModel",
    "CosmosTransformerConfig",
    "CosmosRotaryPosEmbed",
    "MultiViewRoPE",
    "apply_rotary_emb",
    "VideoTokenizer",
    "BaseFlowMatchScheduler",
    "BetaFlowMatchEulerScheduler",
    "BetaSchedulerConfig",
    "FlowMatchUniPCScheduler",
    "SchedulerConfig",
]


def __getattr__(name):  # lazy heavy imports (transformers/peft)
    if name == "CosmosTextEncoder":
        from .text_encoder import CosmosTextEncoder
        return CosmosTextEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
