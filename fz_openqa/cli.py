import hydra
from omegaconf import DictConfig

from fz_openqa.ops.experiment import run_exp
from fz_openqa.ops.tuning import run_tune


@hydra.main(config_path="configs/", config_name="hpo_config.yaml")
def run_hpo(config: DictConfig) -> None:
    # Train model
    return run_tune(config)
