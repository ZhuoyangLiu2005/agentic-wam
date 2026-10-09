"""Hydra search-path plugin: compose configs from configs/ plus its three parts.

configs/
  train.yaml, model/                       shared
  robotwin/{task,data}/                     RoboTwin simulation
  real_pre_train/{task,data}/               real-world pretraining
  real_post_train/{task,data}/              real-world post-training

Adding the three part directories to the search path merges their `task` and `data` groups into one, so
`task=<name>` and `data=<name>` work from every entry point (training, preflight, the RoboTwin eval, the real-robot
server) without naming the part. Hydra discovers this file because `src/` is on sys.path and `hydra_plugins` is a
namespace package. Names must therefore be unique across the parts.
"""
from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

# The configs use AgenticWAM's resolvers (e.g. ${lerobot_shards:<dataset>}); register them for every Hydra
# entry point, including those that never import the trainer (the RoboTwin eval, the real-robot server).
from agenticwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()

CONFIGS = Path(__file__).resolve().parents[3] / "configs"     # <repo>/src/hydra_plugins/agenticwam_configs/<this>
PARTS = ("robotwin", "real_pre_train", "real_post_train")


class AgenticWAMConfigsSearchPath(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        for part in PARTS:
            search_path.append(provider="agenticwam", path=f"file://{CONFIGS / part}")
