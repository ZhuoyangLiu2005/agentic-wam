"""Config loading helpers: YAML -> dict, with `_target_` strings resolved lazily."""
import contextlib
import inspect
import logging
import os
from copy import deepcopy
from typing import Any

import cloudpickle
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from capsim.envs.configs.instantiate import _convert_target_to_string


def _visit_dict_config(cfg: DictConfig | ListConfig | Any, func: Any) -> None:  # type: ignore
    """Apply func recursively to every DictConfig in cfg."""
    if isinstance(cfg, DictConfig):
        func(cfg)
        for v in cfg.values():
            _visit_dict_config(v, func)
    elif isinstance(cfg, ListConfig):
        for v in cfg:
            _visit_dict_config(v, func)


def _cast_to_config(obj: Any) -> Any:
    """Wrap a dict as a DictConfig; anything else is returned unchanged."""
    if isinstance(obj, dict):
        return DictConfig(obj, flags={"allow_objects": True})
    return obj


class DictLoader:
    """Load and save omegaconf configs that may define lazily-constructed objects."""

    @staticmethod
    def load(filenames: str | list[str], keys: None | str | tuple[str, ...] = None) -> Any:
        loaded = DictLoader._load(filenames, keys)
        return OmegaConf.to_container(loaded, resolve=True)  # resolve to basic container

    @staticmethod
    def load_rel(filename: str, keys: None | str | tuple[str, ...] = None) -> Any:
        """Like :meth:`load`, with ``filename`` relative to the caller's source file."""
        caller_frame = inspect.stack()[1]
        caller_fname = caller_frame[0].f_code.co_filename
        assert caller_fname != "<string>", "load_rel Unable to find caller"
        caller_dir = os.path.dirname(caller_fname)
        filename = os.path.join(caller_dir, filename)
        return DictLoader.load(filename, keys)

    @staticmethod
    def _load(filenames: str | list[str], keys: None | str | tuple[str, ...] = None) -> Any:
        """Load one YAML file or merge a list (later files override); optionally return only ``keys``."""
        if not isinstance(filenames, (str, list)):
            raise TypeError("filenames must be a string or a list of strings.")

        has_keys = keys is not None

        def _handle_keys(
            cfg: DictConfig | ListConfig, keys: None | str | tuple[str, ...], has_keys: bool
        ) -> DictConfig | ListConfig | Any | tuple:
            """Return the full config, one key's value, or a tuple of values."""
            if not has_keys:
                return cfg
            if isinstance(keys, str):
                return _cast_to_config(cfg[keys])  # type: ignore
            assert keys is not None
            return tuple(_cast_to_config(cfg[k]) for k in keys)  # type: ignore

        if isinstance(filenames, str):
            filename = os.path.expanduser(filenames).replace("/./", "/")  # redundant
            if os.path.splitext(filename)[1] not in [".yaml", ".yml"]:
                raise ValueError(f"Config file {filename} has to be a yaml file.")

            with open(filename) as f:
                obj = yaml.unsafe_load(f)
            ret = OmegaConf.create(obj)
            return _handle_keys(ret, keys, has_keys)

        elif isinstance(filenames, list):
            if not filenames:
                raise ValueError("No configuration files provided to load.")

            merged_cfg = OmegaConf.create()

            for filename in filenames:
                cfg = DictLoader.load(filename)
                merged_cfg = OmegaConf.merge(merged_cfg, cfg)

            return _handle_keys(merged_cfg, keys, has_keys)

    @staticmethod
    def save(cfg: DictConfig | ListConfig | Any, filename: str) -> None:
        """Save a config to YAML, falling back to ``<filename>.pkl`` when it holds non-serializable objects."""
        logger = logging.getLogger(__name__)
        try:
            cfg = deepcopy(cfg)
        except Exception:
            pass
        else:
            def _replace_type_by_name(x: DictConfig) -> None:
                if "_target_" in x and callable(x._target_):
                    with contextlib.suppress(AttributeError):
                        x._target_ = _convert_target_to_string(x._target_)

            _visit_dict_config(cfg, _replace_type_by_name)

        save_pkl = False
        try:
            dict = OmegaConf.to_container(
                cfg,
                # Keep ${a} interpolations unresolved.
                resolve=False,
            )
            dumped = yaml.dump(dict, default_flow_style=None, allow_unicode=True, width=9999)
            with open(filename, "w") as f:
                f.write(dumped)

            try:
                _ = yaml.unsafe_load(dumped)  # test that it is loadable
            except Exception:
                logger.warning(
                    "The config contains objects that cannot serialize to a valid yaml. "
                    f"{filename} is human-readable but cannot be loaded."
                )
                save_pkl = True
        except Exception:
            logger.exception("Unable to serialize the config to yaml. Error:")
            save_pkl = True

        if save_pkl:
            new_filename = filename + ".pkl"
            try:
                with open(new_filename, "wb") as f:
                    cloudpickle.dump(cfg, f)
                logger.warning(f"Config is saved using cloudpickle at {new_filename}.")
            except Exception:
                pass
