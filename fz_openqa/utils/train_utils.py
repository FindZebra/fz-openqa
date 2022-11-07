import os
from sys import platform
from typing import List

import datasets
import pytorch_lightning as pl
import transformers
from omegaconf import DictConfig
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

import wandb


def empty(*args, **kwargs):
    pass


def silent_huggingface():
    datasets.logging.set_verbosity(datasets.logging.CRITICAL)
    transformers.logging.set_verbosity(transformers.logging.CRITICAL)


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, WandbLogger):
            wandb.finish()


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """
    if trainer.logger is None:
        return

    hparams = {
        k: v
        for k, v in config.items()
        if isinstance(
            v,
            (
                str,
                int,
            ),
        )
    }

    # choose which parts of hydra config will be saved to loggers
    hparams.update(
        {
            k: v
            for k, v in config.items()
            if k
            in (
                "base",
                "trainer",
                "model",
                "datamodule",
                "tokenizer",
                "corpus",
                "callbacks",
            )
        }
    )

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def setup_safe_env():
    os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
    if platform == "darwin":
        # a few flags to fix MacOS stuff
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        import multiprocessing

        try:
            multiprocessing.set_start_method("fork")
        except Exception:
            pass
