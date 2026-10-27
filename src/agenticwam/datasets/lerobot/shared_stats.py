"""Joint normalization statistics for the actual sources in a pretraining mixture.

Every source is transformed to the same action/state coordinates before pooling.
Moments and extrema cover all scanned values. Quantiles use a uniform reservoir
when ``max_samples > 0``; zero keeps every sample for exact pooled quantiles.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np


def training_sources(node) -> Iterator:
    """Walk dataset wrappers/concatenations, excluding unused config definitions."""
    if node.get("datasets") is not None:
        for child in node.datasets:
            yield from training_sources(child)
    elif node.get("dataset") is not None:
        yield from training_sources(node.dataset)
    elif node.get("dataset_dirs") is not None:
        yield node
    else:
        raise ValueError("Shared normalization requires training sources with dataset_dirs")


def validate_shared_normalization(data_cfg) -> Path | None:
    """Reject per-source overrides that would silently split the action space."""
    shared = data_cfg.get("shared_norm_stats")
    if shared is None:
        return None
    path = Path(str(shared)).expanduser().resolve()
    roots = [data_cfg.train]
    if data_cfg.get("val") is not None:
        roots.append(data_cfg.val)
    for source in (source for root in roots for source in training_sources(root)):
        source_path = source.get("pretrained_norm_stats")
        if source_path is None or Path(str(source_path)).expanduser().resolve() != path:
            raise ValueError("Every pretraining source must use data.shared_norm_stats")
        if source.get("save_norm_stats", True):
            raise ValueError("Shared-normalization sources must set save_norm_stats=false")
    if not path.is_file():
        raise FileNotFoundError(
            f"Shared normalization stats not found: {path}. Generate them jointly with "
            "utils/make_stats.py --task <pretraining task> before training."
        )
    return path


class PooledStats:
    """Streaming exact moments plus a reproducible, frame-weighted quantile sample."""

    def __init__(self, max_samples: int = 100_000, seed: int = 42):
        if max_samples < 0:
            raise ValueError("max_samples must be non-negative (0 means exact quantiles)")
        self.capacity = int(max_samples)
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self.sample = None
        self.blocks = []

    def update(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float32)
        if not len(x):
            return
        if not np.isfinite(x).all():
            raise ValueError("Non-finite transformed values in normalization input")
        if self.count and x.shape[1:] != self.mean.shape:
            raise ValueError("Sources have incompatible transformed shapes/horizons")
        mean = x.mean(axis=0, dtype=np.float64)
        m2 = ((x.astype(np.float64) - mean) ** 2).sum(axis=0)
        if not self.count:
            self.mean, self.m2 = mean, m2
            self.minimum, self.maximum = x.min(0), x.max(0)
        else:
            delta = mean - self.mean
            total = self.count + len(x)
            self.m2 += m2 + delta ** 2 * (self.count * len(x) / total)
            self.mean += delta * (len(x) / total)
            self.minimum = np.minimum(self.minimum, x.min(0))
            self.maximum = np.maximum(self.maximum, x.max(0))
        if not self.capacity:
            self.blocks.append(x.copy())
        else:
            if self.sample is None:
                self.sample = np.empty((self.capacity,) + x.shape[1:], dtype=np.float32)
            fill = min(len(x), max(0, self.capacity - self.count))
            if fill:
                self.sample[self.count:self.count + fill] = x[:fill]
            rest = x[fill:]
            if len(rest):
                # Reservoir sampling: item number n replaces a uniformly drawn
                # slot in [0,n). Repeated slots in a batch keep the last write.
                numbers = self.count + fill + np.arange(1, len(rest) + 1)
                slots = (self.rng.random(len(rest)) * numbers).astype(np.int64)
                accepted = np.flatnonzero(slots < self.capacity)
                if len(accepted):
                    _, last = np.unique(slots[accepted][::-1], return_index=True)
                    rows = accepted[::-1][last]
                    self.sample[slots[rows]] = rest[rows]
        self.count += len(x)

    def finish(self) -> dict:
        if not self.count:
            raise ValueError("No valid transformed samples for normalization")
        sample = (np.concatenate(self.blocks, axis=0) if not self.capacity
                  else self.sample[:min(self.count, self.capacity)])
        quantiles = np.percentile(sample, [1, 99], axis=0)
        step = {
            "min": self.minimum, "max": self.maximum,
            "mean": self.mean, "std": np.sqrt(self.m2 / self.count),
            "q01": quantiles[0], "q99": quantiles[1],
        }
        if self.mean.ndim == 1:  # state: one frame, same global and single-step stats
            glob = step
            step = {key: value[None] for key, value in step.items()}
        else:  # action: retain horizon-specific scales and pooled global scales
            global_mean = self.mean.mean(0)
            q = np.percentile(sample.reshape(-1, sample.shape[-1]), [1, 99], axis=0)
            glob = {
                "min": self.minimum.min(0), "max": self.maximum.max(0),
                "mean": global_mean,
                "std": np.sqrt((self.m2 / self.count + (self.mean - global_mean) ** 2).mean(0)),
                "q01": q[0], "q99": q[1],
            }
        return {"default": {
            **{f"global_{key}": value.tolist() for key, value in glob.items()},
            **{f"stepwise_{key}": value.tolist() for key, value in step.items()},
        }}


def _episodes(source):
    """Read numeric columns only; never decode videos or cross episode boundaries."""
    import pyarrow.parquet as pq

    selection = source.get("episode_indices")
    if selection is None and float(source.get("val_set_proportion", 0)) >= 1e-6:
        raise ValueError("Use explicitly selected training shards/episodes for shared stats; "
                         "implicit validation splits are not supported by this offline pass")
    if isinstance(selection, str):
        selection = json.loads(Path(selection).read_text())
    for directory in source.dataset_dirs:
        root = Path(str(directory))
        files = sorted(root.glob("data/chunk-*/*.parquet"))
        if not files:
            raise FileNotFoundError(f"No LeRobot parquet episodes under {root}")
        chosen = selection
        if selection is not None and hasattr(selection, "items"):
            chosen = selection.get(str(directory), selection.get(str(root), []))
        chosen = None if chosen is None else {int(i) for i in chosen}
        for file in files:
            table = pq.read_table(file, columns=["observation.state", "action", "episode_index"])
            episode_ids = table.column("episode_index").to_numpy(zero_copy_only=False).reshape(-1)
            for episode_id in np.unique(episode_ids):
                if chosen is not None and int(episode_id) not in chosen:
                    continue
                rows = np.flatnonzero(episode_ids == episode_id)
                # LeRobot v2 stores one episode per file. Reject fragmented or
                # interleaved episodes rather than constructing misaligned chunks.
                if len(rows) and not np.all(np.diff(rows) == 1):
                    raise ValueError(f"Non-contiguous episode {episode_id} in {file}")
                part = table.slice(int(rows[0]), len(rows))
                state = np.stack(part.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
                action = np.stack(part.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
                if len(state):
                    yield state, action


def compute_shared_stats(data_cfg, *, stride: int = 5, max_samples: int = 100_000,
                         seed: int = 42, batch_windows: int = 256) -> dict:
    """Pool the mixture's transformed values, never average source quantiles.

    All state frames and all action windows selected by ``stride`` contribute to
    exact moments/extrema. Quantiles are exact for max_samples=0, otherwise are
    estimated from a uniform sample over those pooled values, independently of
    which source they came from.
    """
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if stride <= 0 or batch_windows <= 0:
        raise ValueError("stride and batch_windows must be positive")
    states, actions = PooledStats(max_samples, seed), PooledStats(max_samples, seed + 1)
    manifests = []
    horizon = None
    for number, source in enumerate(training_sources(data_cfg.train)):
        proc = source.processor
        if int(proc.action_output_dim) != 64 or int(proc.proprio_output_dim) != 64:
            raise ValueError("Shared pretraining normalization requires unified 64-D action/state")
        if [m.key for m in proc.shape_meta.action] != ["default"] or [m.key for m in proc.shape_meta.state] != ["default"]:
            raise ValueError("Shared stats require one unified action/state key named default")
        steps = int(source.num_frames) - 1
        if horizon is not None and steps != horizon:
            raise ValueError("All sources must have the same action horizon")
        horizon = steps
        frame_stride = int(source.get("global_sample_stride", 1))
        if steps <= 0 or frame_stride <= 0:
            raise ValueError("Action horizon and global_sample_stride must be positive")
        transforms = [instantiate(t) for t in proc.action_state_transforms or []]

        def transform(batch):
            for tf in transforms:
                batch = tf.forward(batch)
            return batch

        frames = episodes = windows = 0
        for st, ac in _episodes(source):
            episodes += 1
            frames += len(st)
            for offset in range(0, len(st), batch_windows):
                state = torch.from_numpy(st[offset:offset + batch_windows, None].copy())
                transformed = transform({"state": {"default": state}})["state"]["default"][:, 0].numpy()
                if transformed.shape[-1] != 64:
                    raise ValueError("Source transform did not produce 64-D states")
                states.update(transformed)
            starts = np.arange(0, max(0, len(st) - (steps - 1) * frame_stride), stride)
            for offset in range(0, len(starts), batch_windows):
                cur = starts[offset:offset + batch_windows]
                indices = cur[:, None] + np.arange(steps)[None] * frame_stride
                batch = {"state": {"default": torch.from_numpy(st[cur, None].copy())},
                         "action": {"default": torch.from_numpy(ac[indices].copy())}}
                transformed = transform(batch)["action"]["default"].numpy()
                if transformed.shape[1:] != (steps, 64):
                    raise ValueError("Source transform did not produce the common action shape")
                actions.update(transformed)
                windows += len(cur)
        if not frames or not windows:
            raise ValueError(f"Training source {number} has no state frames or valid action windows")
        manifests.append({
            "dataset_dirs": [str(p) for p in source.dataset_dirs],
            "transforms": (OmegaConf.to_container(proc.action_state_transforms, resolve=True)
                           if proc.action_state_transforms is not None else []),
            "global_sample_stride": frame_stride,
            "num_episodes": episodes, "num_frames": frames, "num_action_windows": windows,
        })
        print(f"[shared stats] source {number}: {frames} frames, {windows} windows", flush=True)
    return {
        "state": states.finish(), "action": actions.finish(),
        "num_episodes": sum(s["num_episodes"] for s in manifests),
        "num_transition": states.count,
        "shared_normalization": {
            "version": 1, "sources": manifests, "action_horizon": horizon,
            "window_stride": stride, "seed": seed, "max_quantile_samples": max_samples,
            "state_quantiles_exact": not max_samples or states.count <= max_samples,
            "action_quantiles_exact": not max_samples or actions.count <= max_samples,
        },
    }
