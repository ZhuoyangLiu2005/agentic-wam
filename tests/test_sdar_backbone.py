"""Exercise the public SDAR config layout without remote Python or FlashAttention."""
import json

import pytest
import torch

pytest.importorskip("transformers", minversion="4.57")
from transformers import Qwen3Config, Qwen3ForCausalLM
from safetensors.torch import load_file, save_file

from agenticwam.models.experts.code_expert import _load_sdar_backbone


@pytest.fixture
def sdar_checkpoint(tmp_path):
    cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                     num_hidden_layers=1, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=4, tie_word_embeddings=False)
    model = Qwen3ForCausalLM(cfg)
    model.save_pretrained(tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text())
    raw.update(model_type="sdar", architectures=["SDARForCausalLM"],
               auto_map={"AutoModelForCausalLM": "missing_training_module.SDARForCausalLM"},
               fuse_cross_entropy=True)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return tmp_path, model


def test_sdar_qwen3_weight_roundtrip_without_custom_python(sdar_checkpoint):
    path, expected = sdar_checkpoint
    actual = _load_sdar_backbone(path, torch.float32)
    assert isinstance(actual, Qwen3ForCausalLM)
    assert actual.config._attn_implementation == "sdpa"
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["missing", "unexpected"])
def test_sdar_rejects_incomplete_or_incompatible_weights(sdar_checkpoint, corruption):
    path, _ = sdar_checkpoint
    file = path / "model.safetensors"
    weights = load_file(file)
    if corruption == "missing":
        weights.pop("model.layers.0.mlp.up_proj.weight")
    else:
        weights["unknown_module.weight"] = torch.zeros(1)
    save_file(weights, file, metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="SDAR weights do not match"):
        _load_sdar_backbone(path, torch.float32)
