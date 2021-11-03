import os

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from fz_openqa.training.experiment import run_experiment_with_config
from fz_openqa.training.tuning import run_tune_with_config

# OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
# OmegaConf.register_new_resolver("getcwd", os.getcwd)

# todo: even importing this __init__.py will cause the following error: SystemError 15.

# @hydra.main(config_path="configs/", config_name="config.yaml")
# def run_experiment(config: DictConfig):
#     return run_experiment_with_config(config)
#
#
# @hydra.main(config_path="configs/", config_name="hpo_config.yaml")
# def run_tune(config: DictConfig) -> None:
#     return run_tune_with_config(config)


@hydra.main(config_path="configs/", config_name="config.yaml")
def run_experiment(config: DictConfig):
    return run_experiment_with_config(config)
