import os
import re
from typing import Sequence

import omegaconf
from omegaconf import DictConfig

YAML_PATTERN = r"^.*\.yaml$"


def resolve_config_paths(
    config: DictConfig, path: str = "", excludes: Sequence[str] = ["hydra"]
):
    for k, v in ((k_, v_) for k_, v_ in config.items() if k_ not in excludes):
        if isinstance(v, DictConfig):
            resolve_config_paths(v, os.path.join(path, k))
        if isinstance(v, str) and re.findall(YAML_PATTERN, v):
            full_path = os.path.join(path, k, v)
            cfg = omegaconf.OmegaConf.load(full_path)
            config[k] = cfg
