import logging
import os
import shutil
import tempfile
from pathlib import Path

import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from .trainer import AgenticWAMTrainer
from .utils.logging_config import get_logger, setup_logging
from .utils import misc
from .datasets.lerobot.shared_stats import validate_shared_normalization

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def build_datasets(data_cfg: DictConfig):
    shared_stats = validate_shared_normalization(data_cfg)
    if shared_stats is not None and PartialState().is_main_process:
        destination = Path(misc.get_work_dir()) / "dataset_stats.json"
        if destination.resolve() != shared_stats:
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as f:
                    temp_path = Path(f.name)
                shutil.copyfile(shared_stats, temp_path)
                os.replace(temp_path, destination)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
        logger.info("All pretraining sources use shared normalization: %s", shared_stats)
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        if shared_stats is not None:
            return train_ds, instantiate(data_cfg.val)
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def run_training(cfg: DictConfig):
    # Reject missing or inconsistent shared stats before allocating the model.
    validate_shared_normalization(cfg.data)
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data)

    trainer = AgenticWAMTrainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()
