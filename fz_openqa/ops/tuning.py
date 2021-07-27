import os
from typing import Optional

import ray
import rich
from hydra import compose
from hydra import initialize
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from ray.tune import CLIReporter

from fz_openqa.ops.experiment import run_exp
from fz_openqa.utils import train_utils

log = train_utils.get_logger(__name__)

# OmegaConf.register_new_resolver("whoami", lambda: os.environ.get('USER'))


def trial(args, checkpoint_dir=None, **kwargs):
    with initialize(config_path="../configs/"):
        overrides = [f"{k}={v}" for k, v in kwargs.items()]
        overrides += [f"{k}={v}" for k, v in args.items()]
        overrides += [f"work_dir='{os.getcwd()}'"]
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=overrides,
        )
        run_exp(cfg)


def run_tune(config: DictConfig) -> Optional[float]:
    if config.base.print_config:
        print_config(config)

    log.info("Initializing Ray")
    if config.base.server_address is not None:
        ray.util.connect(config.base.server_address)
    else:
        ray.init(**config.ray)

    log.info("Instantiating the search space")
    space = instantiate(config.space)

    log.info("Setting up the trainable")
    trainable = ray.tune.with_parameters(trial, **config.experiment)

    log.info("Instantiating the search algorithm")
    if config.runner.get("search_alg", None) is not None:
        search_alg = instantiate(config.runner.search_alg)
        # cast `points_to_evaluate` into a list
        if search_alg._points_to_evaluate is not None:
            search_alg._points_to_evaluate = [
                p for p in search_alg._points_to_evaluate
            ]
    else:
        search_alg = None

    log.info("Instantiating the runner")
    runner = ray.tune.run(
        trainable,
        config={**space},
        **instantiate(config.runner, search_alg=search_alg),
    )

    log.info("Running HPO!")
    rich.print("Best hyperparameters found were: \n", runner.best_config)


def print_config(config):
    style = "dim"
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)

    for field in config.keys():
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            try:
                branch_content = OmegaConf.to_yaml(
                    config_section, resolve=True
                )
            except Exception:
                pass

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


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
