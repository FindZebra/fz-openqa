import os
from sys import platform
from typing import List, Optional

import rich
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from transformers import PreTrainedTokenizerFast

from fz_openqa.utils import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
    if platform == "darwin":
        # a few flags to fix MacOS stuff
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        import multiprocessing

        multiprocessing.set_start_method("fork")
    if not config.verbose:
        os.environ["WANDB_SILENT"] = "TRUE"

    if config.verbose:
        rich.print(f"> work_dir: {config.work_dir}")
        rich.print(f"> cache_dir: {os.path.abspath(config.cache_dir)}")
        rich.print(f"> Current working directory : {os.getcwd()}")

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init HuggingFace tokenizer
    log.info(f"Instantiating tokenizer <{config.tokenizer._target_}>")
    tokenizer: PreTrainedTokenizerFast = instantiate(config.tokenizer)

    # Init the Corpus
    log.info(
        f"Instantiating corpus <{config.corpus._target_ if config.corpus else 'none'}>"
    )
    corpus: LightningDataModule = (
        instantiate(config.corpus, tokenizer=tokenizer)
        if config.corpus
        else None
    )

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate(
        config.datamodule, tokenizer=tokenizer, corpus=corpus
    )

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = instantiate(
        config.model, tokenizer=tokenizer, corpus=corpus, _recursive_=False
    )

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if config.get("callbacks", None):
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if config.get("logger", None):
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters.")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )
    if (
        trainer.checkpoint_callback
        and trainer.checkpoint_callback.dirpath is not None
    ):
        print(
            f">> checkpoint: {os.path.abspath(trainer.checkpoint_callback.dirpath)}"
        )

    # Train the model
    log.info("Starting training..")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    if not config.trainer.get("fast_dev_run"):
        log.info("Starting testing..")
        trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing..")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(
        f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}"
    )

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
