import os
import socket

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

from fz_openqa import configs
from fz_openqa.training import training
from fz_openqa.utils import train_utils
from fz_openqa.utils.config import print_config
from fz_openqa.utils.config import resolve_config_paths
from fz_openqa.utils.git import git_branch_name
from fz_openqa.utils.git import git_revision_hash
from fz_openqa.utils.git import git_revision_short_hash


def int_div(a, *b):
    y = a
    for x in b:
        y = y / x
    return int(y)


def int_mul(a, *b):
    y = a
    for x in b:
        y *= x
    return int(y)


def int_max(a, *b):
    y = a
    for x in b:
        y = max(x, y)
    return int(y)


def cleanup_bert_name(name):
    return name.split("/")[-1]


N_GPUS = torch.cuda.device_count()
GIT_HASH = git_revision_hash()
GIT_HASH_SHORT = git_revision_short_hash()
GIT_BRANCH_NAME = git_branch_name()
OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))
OmegaConf.register_new_resolver("getcwd", os.getcwd)
OmegaConf.register_new_resolver("hostname", socket.gethostname)
OmegaConf.register_new_resolver("int", lambda x: int(x))
OmegaConf.register_new_resolver("int_mul", int_mul)
OmegaConf.register_new_resolver("int_div", int_div)
OmegaConf.register_new_resolver("int_max", int_max)
OmegaConf.register_new_resolver("n_gpus", lambda *_: N_GPUS)
OmegaConf.register_new_resolver("n_devices", lambda *_: max(1, N_GPUS))
OmegaConf.register_new_resolver("git_hash", lambda *_: GIT_HASH)
OmegaConf.register_new_resolver("git_hash_short", lambda *_: GIT_HASH_SHORT)
OmegaConf.register_new_resolver("eval", lambda x: eval(x))
OmegaConf.register_new_resolver("cleanup_bert_name", cleanup_bert_name)


def run_experiment_with_config(config: DictConfig):
    # replace config paths with loaded configs
    resolve_config_paths(config, path=os.path.dirname(configs.__file__))

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


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def run_experiment(config: DictConfig):
    return run_experiment_with_config(config)
