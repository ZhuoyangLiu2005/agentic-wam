"""Exercise the real deploy adapter's code/text prompt separation without loading weights."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers", minversion="4.57")
from robotwin.deploy.deploy_policy import AgenticWAMCosmosRobotWinPolicy, DEFAULT_PROMPT, _CosmosTextEncoder


@pytest.mark.parametrize("task_assets", [False, True])
def test_stage_code_gets_raw_instruction_and_wam_keeps_skill_prompt(monkeypatch, tmp_path, task_assets):
    for key in list(__import__("os").environ):
        if key.startswith("AGENTICWAM_"):
            monkeypatch.delenv(key)
    if task_assets:
        p = tmp_path / "preamble.txt"
        p.write_text("def press(): ...\n")
        monkeypatch.setenv("AGENTICWAM_CODE_PREAMBLE_FILE", str(p))
    captured = {}
    def infer(**kwargs):
        captured["code_instruction"] = kwargs["instruction"]
        return torch.zeros(1, 32, 14), "# [stage 1/1]\npress()"
    def encode(prompt):
        captured["text_instruction"] = prompt
        return torch.zeros(1, 1, 1)
    policy = AgenticWAMCosmosRobotWinPolicy.__new__(AgenticWAMCosmosRobotWinPolicy)
    policy.model = SimpleNamespace(use_code_expert=True, infer_action_trimodal=infer)
    policy.device = "cpu"
    policy._build_obs_video = lambda observation: torch.zeros(1, 3, 3, 2, 2)
    policy._normalize_state = lambda state: torch.zeros(1, 14)
    policy._canonical_instruction = lambda instruction: instruction
    policy._skill_prompt = lambda: "SKILL LIBRARY\n"
    policy.text = SimpleNamespace(encode=encode)
    policy.video_log_dir = None
    policy._save_all_replans = False
    policy.step_count = 1
    policy.action_horizon = 32
    policy.num_video_frames = 9
    policy.num_inference_steps = 20
    policy.sigma_shift = 5.0
    policy.code_cache = {}
    policy._denormalize_action = lambda action: np.zeros((1, 32, 14), dtype=np.float32)
    instruction = "press the alarm clock"
    policy._infer_action_chunk({"joint_action": {"vector": np.zeros(14)}}, instruction)
    wrapped = DEFAULT_PROMPT.format(task=instruction)
    assert captured["text_instruction"] == wrapped
    assert captured["code_instruction"] == (instruction if task_assets else "SKILL LIBRARY\n" + wrapped)


@pytest.mark.parametrize("offload", [False, True])
def test_text_encoder_offload_keeps_cached_embeddings(offload):
    moves = []
    forwards = []
    class Encoder:
        def to(self, device):
            moves.append(device)
            return self
        def __call__(self, ids, **kwargs):
            forwards.append(ids.clone())
            hidden = torch.tensor([[[1., 2., 4.], [2., 5., 8.]]])
            return SimpleNamespace(hidden_states=[hidden * 0, hidden])
    encoder = _CosmosTextEncoder.__new__(_CosmosTextEncoder)
    encoder.qwen = Encoder()
    encoder.proc = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1, 2])
    encoder.device = "cpu"
    encoder.max_len = 2
    encoder.offload = offload
    encoder._cache = {}
    first = encoder.encode("press the clock")
    assert first.shape == (1, 2, 3)
    assert float(first.std()) > 0
    assert encoder.encode("press the clock") is first
    assert len(forwards) == 1
    assert moves == (["cpu", "cpu"] if offload else [])
