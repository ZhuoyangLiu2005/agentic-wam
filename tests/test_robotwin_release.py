"""CPU regressions for the released checkpoints' prompts, canvas and mode selection."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "robotwin"))
from hybrid_eval import config, launch


@pytest.fixture
def release_site(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "configs/robotwin"
    (cfg_dir / "models").mkdir(parents=True)
    for name in ("robotwin_wam", "robotwin_hybrid"):
        (cfg_dir / "models" / (name + ".yaml")).write_text(
            (ROOT / "configs/robotwin/models" / (name + ".yaml")).read_text())
    for family, canvas in (("tasks", 112), ("tasks_indep", 72)):
        task = cfg_dir / family / "click_alarmclock"
        task.mkdir(parents=True)
        (task / "preamble.txt").write_text("def reach(): ...\n")
        (task / "frozen_program.py").write_text("def reach(): pass\n")
        (task / "task.yaml").write_text(f"gen_tokens: {canvas}\n")
    bundle = tmp_path / "checkpoints/robotwin_release"
    for family, filename in (("wam", "step_200000.pt"), ("hybrid", "step_180000.pt")):
        (bundle / family).mkdir(parents=True)
        (bundle / family / filename).write_bytes(b"fixture")
    (bundle / "dataset_stats.json").write_text("{}")
    (bundle / "task_skill_prompt.json").write_text("{}")
    raw = yaml.safe_load((ROOT / "configs/robotwin/site.example.yaml").read_text())
    raw["repo"] = str(tmp_path)
    raw["python_bin"] = str(Path(sys.executable).parent)
    site = config._resolve(raw, raw)
    monkeypatch.setattr(config, "REPO", tmp_path)
    monkeypatch.setattr(config, "CFG_DIR", cfg_dir)
    monkeypatch.setattr(config, "site", lambda: site)
    monkeypatch.setattr(launch, "REPO", tmp_path)
    for key in list(os.environ):
        if key.startswith("AGENTICWAM_"):
            monkeypatch.delenv(key)
    # A fixed-canvas preset must not consult a private training corpus/tokenizer.
    def no_measurement(*args):
        raise AssertionError("fixed release canvas unexpectedly measured from training data")
    monkeypatch.setattr(launch, "measured_L", no_measurement)
    return site, bundle


def plan(model, mode="wam", extra=()):
    return launch.plan(launch.parse_args([
        "--model", model, "--mode", mode, "--task", "click_alarmclock", "--shard-gpu", "0", *extra]))


def test_wam_uses_episode_instruction_and_crop_and_clears_hybrid_state(release_site, monkeypatch):
    monkeypatch.setenv("TASK", "a_different_task")
    monkeypatch.setenv("AGENTICWAM_HYBRID", "1")
    monkeypatch.setenv("AGENTICWAM_TASK_INSTRUCTION_MAP", "/stale/canonical.json")
    monkeypatch.setenv("AGENTICWAM_CODE_PREAMBLE_FILE", "/stale/preamble.txt")
    p = plan("robotwin_wam")
    assert p["meta"]["ckpt"].endswith("/wam/step_200000.pt")
    assert p["meta"]["instruction_type"] == "seen"
    assert p["env"]["TASK"] == "click_alarmclock"
    assert p["meta"]["gen_tokens"] == 128
    assert p["env"]["AGENTICWAM_HYBRID"] == "0"
    assert p["env"]["AGENTICWAM_TASK_INSTRUCTION_MAP"] == ""
    assert "AGENTICWAM_CODE_PREAMBLE_FILE" not in p["env"]
    assert p["env"]["AGENTICWAM_EVAL_PREPROC"] == "parity_crop"
    assert p["meta"]["eval_preproc_crop"] == 0.95
    assert "EVALUATION.instruction_type=seen" in p["cmd"]


@pytest.mark.parametrize("mode", ["wam", "hybrid"])
def test_hybrid_checkpoint_uses_independent_assets_and_fixed_canvas_in_both_modes(release_site, mode):
    p = plan("robotwin_hybrid", mode)
    assert p["meta"]["ckpt"].endswith("/hybrid/step_180000.pt")
    assert p["task"]["preamble"].endswith("/tasks_indep/click_alarmclock/preamble.txt")
    assert p["meta"]["gen_tokens"] == 124
    assert p["meta"]["nis"] == 40
    assert p["meta"]["replan"] == 24
    assert p["env"]["AGENTICWAM_CODE_NO_HISTORY"] == "1"
    assert p["env"]["AGENTICWAM_HYBRID"] == ("1" if mode == "hybrid" else "0")
    assert p["env"]["AGENTICWAM_HYBRID_TASKS"].endswith("/tasks_indep")
    assert p["meta"]["decode"] == {"conf_thresh": 0.5, "reveal_cap": 4, "nn_repair": False}


def test_explicit_checkpoint_canvas_and_instruction_override_preset(release_site):
    _, bundle = release_site
    other = bundle / "another.pt"
    other.write_bytes(b"other")
    p = plan("robotwin_hybrid", "hybrid", (
        "--ckpt", str(other), "--gen-tokens", "120", "--instruction-type", "unseen"))
    assert p["meta"]["ckpt"] == str(other)
    assert p["meta"]["gen_tokens"] == 120
    assert p["meta"]["instruction_type"] == "unseen"


def test_legacy_training_run_preserves_milestone_fallback(tmp_path):
    milestone = tmp_path / "milestones/step_040000.pt"
    milestone.parent.mkdir()
    milestone.write_bytes(b"weights")
    assert config.resolve_checkpoint({"run": str(tmp_path)}, step=40000) == str(milestone)


def test_missing_release_checkpoint_reports_download_command(release_site):
    _, bundle = release_site
    (bundle / "wam/step_200000.pt").unlink()
    with pytest.raises(SystemExit, match="hf download miniFranka/agentic-wam-posttrain-robotwin-suit"):
        plan("robotwin_wam")


def test_wam_only_checkpoint_rejects_hybrid(release_site):
    with pytest.raises(SystemExit, match="task_assets: false"):
        plan("robotwin_wam", "hybrid")


def test_wrapper_dry_run_never_starts_perception(release_site, monkeypatch):
    from hybrid_eval import vision
    spec = importlib.util.spec_from_file_location("release_eval_wrapper", ROOT / "scripts/eval_robotwin_release.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run started perception")
    monkeypatch.setattr(vision, "ensure_up", forbidden)
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(wrapper.subprocess, "run", run)
    assert wrapper.main(["hybrid", "--vis-gpu", "1", "--dry-run", "--plan", "smoke"]) == 0
    assert "--dry-run" in calls[0]
    assert "robotwin_hybrid" in calls[0]


@pytest.mark.parametrize("token", [None, "fixture-token"])
def test_sam3_start_accepts_env_or_cached_login_without_token_file(release_site, monkeypatch, token):
    from hybrid_eval import vision
    site, _ = release_site
    site["hf_token_file"] = "/missing/legacy-token-file"
    if token:
        monkeypatch.setenv("HF_TOKEN", token)
    else:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(vision, "status", lambda _: {"molmo": True, "sam3": False, "graspnet": True})
    envs = []
    monkeypatch.setattr(vision, "_spawn", lambda cmd, cwd, env, log: envs.append(env))
    vision.start(site, gpu=1, say=lambda _: None)
    assert envs[0].get("HF_TOKEN") == token


def test_sam3_start_retains_optional_token_file(release_site, monkeypatch, tmp_path):
    from hybrid_eval import vision
    site, _ = release_site
    token_file = tmp_path / "token"
    token_file.write_text("fixture-token\n")
    site["hf_token_file"] = str(token_file)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(vision, "status", lambda _: {"molmo": True, "sam3": False, "graspnet": True})
    envs = []
    monkeypatch.setattr(vision, "_spawn", lambda cmd, cwd, env, log: envs.append(env))
    vision.start(site, gpu=1, say=lambda _: None)
    assert envs[0]["HF_TOKEN"] == "fixture-token"


@pytest.mark.parametrize("override", [False, True])
def test_molmo_start_bounds_multimodal_profiling(release_site, monkeypatch, override):
    import json
    from hybrid_eval import vision
    site, _ = release_site
    if override:
        site["vision"]["molmo"].update(max_num_seqs=2, limit_mm_per_prompt={"image": 2, "video": 0})
    monkeypatch.setattr(vision, "status", lambda _: {"molmo": False, "sam3": True, "graspnet": True})
    calls = []
    monkeypatch.setattr(vision, "_spawn", lambda cmd, cwd, env, log: calls.append(cmd))
    vision.start(site, gpu=1, say=lambda _: None)
    cmd = calls[0]
    assert cmd[cmd.index("--max-num-seqs") + 1] == ("2" if override else "1")
    assert json.loads(cmd[cmd.index("--limit-mm-per-prompt") + 1]) == {"image": 2 if override else 1, "video": 0}


@pytest.mark.parametrize("success,total,rc,expected", [
    (1, 1, 0, (1, 1)),
    (0, 1, 0, (0, 1)),
    (1, 1, 1, None),
    (4, 4, 1, None),
    (4, 4, 0, None),
    (0, 0, 0, None),
])
def test_benchmark_resumes_only_complete_successful_processes(tmp_path, success, total, rc, expected):
    import json
    from robotwin.eval.benchmark import job_result
    (tmp_path / "run_meta.json").write_text(json.dumps({
        "result": {"success": success, "total": total, "rc": rc}}))
    assert job_result({"results_dir": str(tmp_path), "episodes": 1}) == expected
