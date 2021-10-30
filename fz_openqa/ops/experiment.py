import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from omegaconf import OmegaConf

from fz_openqa import configs
from fz_openqa.ops import training
from fz_openqa.utils import train_utils
from fz_openqa.utils.config import resolve_config_paths

OmegaConf.register_new_resolver("getcwd", os.getcwd)
OmegaConf.register_new_resolver("get_original_cwd", get_original_cwd)
OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))


@hydra.main(config_path="../configs/", config_name="config.yaml")
def run_experiment(config: DictConfig):
    # todo: try re-inserrting local imakemports again
    # from fz_openqa.ops import training
    # from fz_openqa.utils import train_utils

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
