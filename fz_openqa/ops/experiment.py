import os
from typing import Dict

import hydra
from omegaconf import DictConfig

from fz_openqa import configs
from fz_openqa.utils.config import resolve_config_paths


@hydra.main(
    config_path="../configs/",
    config_name="config.yaml",
)
def run_exp(config: DictConfig) -> Dict[str, float]:
    return _run_exp(config)


def _run_exp(config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from fz_openqa.ops import training
    from fz_openqa.utils import train_utils

    # replace config paths with loaded configs
    resolve_config_paths(config, path=os.path.dirname(configs.__file__))

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # - forcing multi-gpu friendly configuration
    # You can safely get rid of this line if you don't want those
    train_utils.extras(config)
    # Pretty print config using Rich library
    if config.get("print_config"):
        train_utils.print_config(config, resolve=True)

    # Train model
    return training.train(config)
