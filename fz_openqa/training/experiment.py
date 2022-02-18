import os

import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf

from fz_openqa import configs
from fz_openqa.training import training
from fz_openqa.utils import train_utils
from fz_openqa.utils.config import print_config
from fz_openqa.utils.config import resolve_config_paths

OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)
OmegaConf.register_new_resolver("int_mul", lambda x, y: int(x * y))
OmegaConf.register_new_resolver("int_div", lambda x, y: int(x / y))


def run_experiment_with_config(config: DictConfig):
    # replace config paths with loaded configs
    resolve_config_paths(config, path=os.path.dirname(configs.__file__))

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # - forcing multi-gpu friendly configuration
    # You can safely get rid of this line if you don't want those
    train_utils.extras(config)

    # set random seed if not specified
    if config.base.get("seed", None) is None:
        config.base.seed = int(np.random.randint(0, 2 ** 32 - 1))

    # save config to file
    with open("config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(config))

    # Pretty print config using Rich library
    if config.get("print_config"):
        print_config(config, resolve=False)

    # Train model
    return training.train(config)


@hydra.main(config_path="../configs/", config_name="config.yaml")
def run_experiment(config: DictConfig):
    return run_experiment_with_config(config)
