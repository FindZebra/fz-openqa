import os
from pathlib import Path
from typing import List
from typing import Optional

import rich
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from transformers import PreTrainedTokenizerFast

import fz_openqa
from fz_openqa.utils import train_utils
from fz_openqa.utils.train_utils import setup_safe_env

log = train_utils.get_logger(__name__)

_root = Path(fz_openqa.__file__).parent.parent

OmegaConf.register_new_resolver("getcwd", lambda: os.getcwd())
OmegaConf.register_new_resolver("get_original_cwd", lambda: _root)
OmegaConf.register_new_resolver("whoami", lambda: os.environ.get("USER"))


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    setup_safe_env()

    if not config.verbose:
        os.environ["WANDB_SILENT"] = "TRUE"

    if config.verbose:
        rich.print(f"> work_dir: {config.work_dir}")
        rich.print(f"> original_wdir: {config.original_wdir}")
        rich.print(f"> cache_dir: {os.path.abspath(config.cache_dir)}")
        rich.print(f"> Current working directory : {os.getcwd()}")

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init HuggingFace tokenizer
    log.info(f"Instantiating tokenizer <{config.tokenizer._target_}>")
    tokenizer: PreTrainedTokenizerFast = instantiate(config.tokenizer)

    # Init the Corpus
    has_corpus = config.corpus and "_target_" in config.corpus.keys()
    log.info(
        f"Instantiating corpus <{config.corpus._target_ if has_corpus else 'none'}>"
    )
    corpus: LightningDataModule = (
        instantiate(config.corpus, tokenizer=tokenizer) if has_corpus else None
    )

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = instantiate(
        config.datamodule,
        tokenizer=tokenizer,
        corpus=corpus,
        _recursive_=False,
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
    train_utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
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
    train_utils.finish(
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
