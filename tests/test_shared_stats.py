"""CPU regressions for pooled multi-source pretraining normalization."""
import ast
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from omegaconf import DictConfig, OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agenticwam.datasets.lerobot.shared_stats import (
    PooledStats, compute_shared_stats, training_sources, validate_shared_normalization,
)


ROOT = Path(__file__).resolve().parents[1]


def source(root, raw_dim, transform, stats_path):
    return {
        "dataset_dirs": [str(root)], "num_frames": 4,
        "global_sample_stride": 1, "val_set_proportion": 0,
        "pretrained_norm_stats": str(stats_path), "save_norm_stats": False,
        "processor": {
            "action_output_dim": 64, "proprio_output_dim": 64,
            "shape_meta": {"action": [{"key": "default", "raw_shape": raw_dim, "shape": 64}],
                           "state": [{"key": "default", "raw_shape": raw_dim, "shape": 64}]},
            "action_state_transforms": [{
                "_target_": "agenticwam.datasets.lerobot.transforms." + transform,
                "keys": ["default"],
            }],
        },
    }


def write_episode(root, raw_dim, length, origin, speed, episode=0):
    state = np.zeros((length, raw_dim), dtype=np.float32)
    for start in (0, 16):
        pose = np.broadcast_to(np.eye(4, dtype=np.float32), (length, 4, 4)).copy()
        pose[:, 0, 3] = origin + np.arange(length) * speed
        state[:, start:start + 16] = pose.reshape(length, 16)
    if raw_dim == 82:
        state[:, 32:34] = .5
    else:
        state[:, 32:] = .25
    action = state[np.minimum(np.arange(length) + 1, length - 1)].copy()
    table = pa.table({"observation.state": state.tolist(), "action": action.tolist(),
                      "episode_index": [episode] * length})
    directory = root / "data/chunk-001"
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, directory / f"episode_{episode:06d}.parquet")


@pytest.fixture
def mixture(tmp_path):
    stats = tmp_path / "joint.json"
    yam, ego = tmp_path / "yam", tmp_path / "ego"
    write_episode(yam, 82, 7, 0, 1)
    write_episode(ego, 76, 13, 100, 10)
    a = source(yam, 82, "yam_unified_eef.YamUnifiedEEFTransform", stats)
    b = source(ego, 76, "ego_unified_eef.EgoUnifiedEEFTransform", stats)
    return OmegaConf.create({
        "shared_norm_stats": str(stats),
        "train": {"dataset": {"datasets": [a, b]}},
    })


def test_joint_quantiles_and_moments_are_pooled_not_averaged():
    a = np.arange(200, dtype=np.float32).reshape(100, 2)
    b = np.array([[1000, 2000]], dtype=np.float32)
    acc = PooledStats(max_samples=0)
    acc.update(a[:60]); acc.update(a[60:]); acc.update(b)
    got = acc.finish()["default"]
    pooled = np.concatenate([a, b])
    for name, fn in [("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)]:
        np.testing.assert_allclose(got["global_" + name], fn(pooled.astype(np.float64), axis=0))
    np.testing.assert_allclose(got["global_q99"], np.percentile(pooled, 99, axis=0))
    assert not np.allclose(got["global_q99"], (np.percentile(a, 99, axis=0) + b[0]) / 2)


def test_bounded_quantiles_keep_exact_full_population_moments():
    x = np.arange(1200, dtype=np.float32).reshape(100, 3, 4)
    acc, same = PooledStats(7, 9), PooledStats(7, 9)
    for block in np.array_split(x, 8):
        acc.update(block); same.update(block)
    got = acc.finish()["default"]
    assert acc.count == 100 and acc.sample.shape == (7, 3, 4)
    np.testing.assert_array_equal(acc.sample, same.sample)
    np.testing.assert_allclose(got["stepwise_mean"], x.mean(0))
    np.testing.assert_allclose(got["stepwise_std"], x.astype(np.float64).std(0))
    np.testing.assert_allclose(got["global_std"], x.reshape(-1, 4).astype(np.float64).std(0))
    np.testing.assert_array_equal(got["global_min"], x.min((0, 1)))
    np.testing.assert_array_equal(got["global_max"], x.max((0, 1)))


def test_real_yam_and_ego_transforms_contribute_to_one_distribution(mixture):
    got = compute_shared_stats(mixture, stride=1, max_samples=0)
    state_x = np.r_[np.arange(7), 100 + np.arange(13) * 10]
    np.testing.assert_allclose(got["state"]["default"]["global_mean"][0], state_x.mean())
    np.testing.assert_allclose(got["state"]["default"]["global_q99"][0], np.percentile(state_x, 99))
    action_x = []
    for length, speed in [(7, 1), (13, 10)]:
        for start in range(length - 2):
            action_x.append([(min(start + step + 1, length - 1) - start) * speed for step in range(3)])
    action_x = np.asarray(action_x)
    np.testing.assert_allclose(np.asarray(got["action"]["default"]["stepwise_mean"])[:, 0], action_x.mean(0))
    np.testing.assert_allclose(np.asarray(got["action"]["default"]["stepwise_q99"])[:, 0], np.percentile(action_x, 99, axis=0))
    assert got["num_transition"] == 20 and got["num_episodes"] == 2
    assert len(got["shared_normalization"]["sources"]) == 2
    assert got["shared_normalization"]["action_quantiles_exact"]


def test_source_selection_and_new_sources_are_not_hardcoded(mixture):
    nodes = list(training_sources(mixture.train))
    root = Path(nodes[0].dataset_dirs[0])
    write_episode(root, 82, 5, 10000, 3, episode=1)
    nodes[0].episode_indices = [0]
    nodes[0].val_set_proportion = .2  # explicit selection overrides the implicit split, as in training
    mixture.train.dataset.datasets.append(OmegaConf.to_container(nodes[1]))
    got = compute_shared_stats(mixture, stride=1, max_samples=0)
    assert got["num_transition"] == 7 + 13 + 13
    assert len(got["shared_normalization"]["sources"]) == 3
    assert got["state"]["default"]["global_max"][0] == 220


def test_missing_or_per_source_stats_are_rejected(mixture):
    with pytest.raises(FileNotFoundError, match="Generate them jointly"):
        validate_shared_normalization(mixture)
    path = Path(mixture.shared_norm_stats)
    path.write_text("{}")
    assert validate_shared_normalization(mixture) == path
    mixture.train.dataset.datasets[1].pretrained_norm_stats = "different.json"
    with pytest.raises(ValueError, match="Every pretraining source"):
        validate_shared_normalization(mixture)


def test_runtime_persists_shared_stats_once(mixture, tmp_path):
    path = Path(mixture.shared_norm_stats)
    path.write_text(json.dumps(compute_shared_stats(mixture, stride=1, max_samples=0)))
    output = tmp_path / "run"; output.mkdir()
    # Exercise the real runtime function without constructing a model or loading videos.
    tree = ast.parse((ROOT / "src/agenticwam/runtime.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_datasets")
    copyfile = Mock(wraps=shutil.copyfile)
    instance = object()
    ns = {"DictConfig": DictConfig, "validate_shared_normalization": validate_shared_normalization,
          "PartialState": lambda: SimpleNamespace(is_main_process=True),
          "Path": Path, "tempfile": tempfile, "os": os,
          "shutil": SimpleNamespace(copyfile=copyfile),
          "misc": SimpleNamespace(get_work_dir=lambda: str(output)),
          "logger": logging.getLogger(__name__), "instantiate": lambda cfg: instance}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "runtime.py", "exec"), ns)
    assert ns["build_datasets"](mixture) == (instance, instance)
    assert copyfile.call_count == 1
    assert (output / "dataset_stats.json").read_bytes() == path.read_bytes()
    mixture.shared_norm_stats = str(output / "dataset_stats.json")
    for node in training_sources(mixture.train):
        node.pretrained_norm_stats = mixture.shared_norm_stats
    ns["build_datasets"](mixture)
    assert copyfile.call_count == 1  # already canonical: no second stats file/write


def test_empty_nonfinite_and_incompatible_inputs_fail():
    acc = PooledStats(0)
    with pytest.raises(ValueError, match="No valid"):
        acc.finish()
    with pytest.raises(ValueError, match="Non-finite"):
        acc.update(np.array([[np.nan]], dtype=np.float32))
    acc.update(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="incompatible"):
        acc.update(np.zeros((2, 4), dtype=np.float32))


def test_mismatched_horizons_and_implicit_validation_are_rejected(mixture):
    mixture.train.dataset.datasets[1].num_frames = 5
    with pytest.raises(ValueError, match="same action horizon"):
        compute_shared_stats(mixture, max_samples=10)
    mixture.train.dataset.datasets[1].num_frames = 4
    mixture.train.dataset.datasets[1].val_set_proportion = .2
    with pytest.raises(ValueError, match="validation splits"):
        compute_shared_stats(mixture, max_samples=10)


def test_pretraining_config_and_joint_cli(tmp_path):
    from hydra import compose, initialize_config_dir
    from agenticwam.utils.config_resolvers import register_default_resolvers

    yam, ego = tmp_path / "yam", tmp_path / "ego"
    write_episode(yam, 82, 40, 0, 1)
    write_episode(ego, 76, 44, 100, 2)
    stats = tmp_path / "mixture/dataset_stats_eef64.json"
    overrides = [f"data.yam_root={yam}", f"data.ego_root={ego}", f"data.shared_norm_stats={stats}"]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, str(ROOT / "utils/make_stats.py"), "--task", "pretrain_magic_soup",
         "--stride", "1", "--max-samples", "0", *overrides],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=35,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STATS_DONE" in result.stdout
    payload = json.loads(stats.read_text())
    assert payload["num_transition"] == 84
    assert len(payload["action"]["default"]["stepwise_mean"]) == 32
    assert list(stats.parent.glob("*.json")) == [stats]
    assert not list(yam.glob("*stats*.json")) and not list(ego.glob("*stats*.json"))
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=pretrain_magic_soup", *overrides])
        assert validate_shared_normalization(cfg.data) == stats
        for node in training_sources(cfg.data.train):
            assert node.pretrained_norm_stats == str(stats)
            assert node.save_norm_stats is False
