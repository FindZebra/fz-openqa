import os
import sys
from copy import copy

import torch
from datasets import Split

from fz_openqa.training.training import load_checkpoint
from fz_openqa.utils.elasticsearch import ElasticSearchInstance
from fz_openqa.utils.fingerprint import get_fingerprint

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import logging
from pathlib import Path
from typing import Optional

import datasets
import hydra
import rich
import transformers
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer

import fz_openqa
from fz_openqa.modeling.zero_shot import ZeroShot
from fz_openqa import configs
from fz_openqa.datamodules.builders import QaBuilder, ConcatQaBuilder
from fz_openqa.datamodules.builders import CorpusBuilder
from fz_openqa.datamodules.builders import OpenQaBuilder
from fz_openqa.datamodules.datamodule import DataModule
from fz_openqa.datamodules.index.builder import IndexBuilder
from fz_openqa.datamodules.pipes import ExactMatch, PrioritySampler
from fz_openqa.inference.checkpoint import CheckpointLoader
from fz_openqa.tokenizers.pretrained import init_pretrained_tokenizer
from fz_openqa.utils.config import print_config

logger = logging.getLogger(__name__)

import fz_openqa.training.experiment  # type: ignore


@hydra.main(
    config_path=str(Path(configs.__file__).parent),
    config_name="eval_config.yaml",
)
def run(config):
    """Load the OpenQA dataset mapped using a pre-trained model.

    On the cluster, run:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 poetry run python \
    examples/load_openqa.py \
    sys=titan trainer.strategy=dp trainer.gpus=8 +batch_size=1000 +num_workers=10
    ```
    """
    print_config(config)
    # set the context
    datasets.set_caching_enabled(True)
    # datasets.logging.set_verbosity(datasets.logging.CRITICAL)
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(1, workers=True)

    # load the model
    overrides = DictConfig(OmegaConf.to_container(config.overrides, resolve=True))
    logger.info("overriding config with:")
    print_config(overrides)
    checkpoint_manager = load_checkpoint(
        config.checkpoint,
        override_config=overrides,
        ref_config=config,
        silent=True,
    )
    model = checkpoint_manager.load_model(config.checkpoint_type)
    checkpoint_config = checkpoint_manager.config
    logger.info("Checkpoint config:")
    print_config(checkpoint_config)
    logger.info(f"Model fingerprint: {get_fingerprint(model)}")

    # load the trainer and dataset
    trainer: Trainer = instantiate(checkpoint_config.trainer)
    # instantiate the datamodule
    logger.info(f"Instantiating datamodule <{checkpoint_config.datamodule._target_}>")
    datamodule: DataModule = instantiate(checkpoint_config.datamodule)
    with ElasticSearchInstance(stdout=open("es.stdout.log", "w")):
        datamodule.setup(trainer=trainer, model=model, clean_caches=False)
    if config.verbose:
        rich.print(datamodule.dataset)
        datamodule.display_samples(n_samples=1, split=config.split)

    # evaluate the model
    logits_key = "_reader_logits_"
    all_preds = []
    for n in range(config.n_samples):
        model.tracked_metrics = {logits_key: None}
        loader = {
            Split.VALIDATION: datamodule.val_dataloader,
            Split.TEST: datamodule.test_dataloader,
        }[config.split]()
        output = trainer.test(model=model, dataloaders=loader, verbose=True)
        rich.print(f"=== Sample {n} ===")
        rich.print(output)
        # store predictions
        rich.print(model.tracked_metrics)
        all_preds.append(model.tracked_metrics[logits_key])
    probs = torch.stack(all_preds, dim=0)
    probs = probs.float().softmax(dim=-1).mean(0)
    rich.print(probs[:10])
    preds = probs.argmax(dim=-1)
    rich.print(f"# probs.shape: {probs.shape}")
    targets = datamodule.dataset[config.split]["answer.target"]
    rich.print(f"# targets.shape: {targets.shape}")

    acc = (preds == targets).float().mean()
    rich.print(f"# Accuracy (n={config.n_samples}): {acc:.2%}")


if __name__ == "__main__":
    run()
