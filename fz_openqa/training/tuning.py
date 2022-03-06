import os

import hydra
import ray
import rich
from hydra import compose
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from fz_openqa.training.experiment import run_experiment_with_config
from fz_openqa.utils import train_utils
from fz_openqa.utils.config import print_config

log = train_utils.get_logger(__name__)

# OmegaConf.register_new_resolver("whoami", lambda: os.environ.get('USER'))

DEFAULT_GLOBALS = ["experiment", "environ"]


def format_key(k, globals=DEFAULT_GLOBALS):
    if k in globals:
        return f"+{k}"
    else:
        return k


def trial(args, checkpoint_dir=None, **kwargs):
    if not GlobalHydra.instance().is_initialized():
        initialize(config_path="../configs/")

    overrides = [f"{format_key(k)}={v}" for k, v in kwargs.items()]
    overrides += [f"{format_key(k)}={v}" for k, v in args.items()]
    overrides += [f"sys.work_dir='{os.getcwd()}'"]
    cfg = compose(
        config_name="config.yaml",
        return_hydra_config=True,
        overrides=overrides,
    )
    run_experiment_with_config(cfg)


def run_tune_with_config(config: DictConfig):
    # todo: error: SystemError 1: debug by runnning simple code in `trial`...
    if config.get("print_config", True):
        print_config(config, resolve=True)

    log.info("Initializing Ray")
    if config.base.server_address is not None:
        ray.util.connect(config.base.server_address)
    else:
        cfg = dict(config.ray)
        pwd = cfg.pop("_redis_password", None)
        pwd = str(pwd) if pwd is not None else None
        ray.init(_redis_password=pwd, **cfg)

    log.info("Instantiating the search space")
    space = instantiate(config.space)

    log.info("Setting up the trainable")
    trainable = ray.tune.with_parameters(trial, **config.experiment)

    log.info("Instantiating the search algorithm")
    if config.runner.get("search_alg", None) is not None:
        search_alg = instantiate(config.runner.search_alg)
        # cast `points_to_evaluate` into a list
        if search_alg._points_to_evaluate is not None:
            search_alg._points_to_evaluate = [p for p in search_alg._points_to_evaluate]
    else:
        search_alg = None

    log.info("Instantiating the runner")
    runner = ray.tune.run(
        trainable,
        config=OmegaConf.to_object(space),
        **instantiate(config.runner, search_alg=search_alg),
    )

    log.info("Running HPO!")
    rich.print("Best hyperparameters found were: \n", runner.best_config)


@hydra.main(config_path="../configs/", config_name="hpo_config.yaml")
def run_tune(config: DictConfig) -> None:
    return run_tune_with_config(config)


if __name__ == "__main__":
    args = {
        "trainer.checkpoint_callback": True,
        "callbacks": "default",
        "experiment": "sandbox",
        "model.head.hidden_size": 32,
        "work_dir": "/Users/valv/Documents/Research/code/_main/fz-ner/mutliruns/",
        "cache_dir": "/Users/valv/Documents/Research/code/_main/fz-ner/cache/",
    }
    trial(args)
