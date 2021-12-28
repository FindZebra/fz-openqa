import logging
import os
import threading
import time
import warnings
from functools import partial
from typing import List
from typing import Optional

import datasets
import rich
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from fz_openqa.callbacks.index_openqa import IndexOpenQaCallback
from fz_openqa.datamodules import DataModule
from fz_openqa.utils import train_utils
from fz_openqa.utils.pretty import pprint_batch
from fz_openqa.utils.train_utils import setup_safe_env

log = train_utils.get_logger(__name__)


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
        # datasets.logging.set_verbosity(datasets.logging.ERROR)
        # datasets.disable_progress_bar()

    # set verbosity
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    datasets.logging.set_verbosity(datasets.logging.CRITICAL)
    # avoid "too many open files" error
    torch.multiprocessing.set_sharing_strategy("file_system")

    log.info(f"work_dir={config.sys.work_dir}")
    log.info(f"cache_dir={os.path.abspath(config.sys.cache_dir)}")
    log.info(f"Experiment working directory: {os.getcwd()}")

    # # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.base.seed, workers=True)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: DataModule = instantiate(config.datamodule)

    # only preprocess the data if there is no trainer
    if config.get("trainer", None) is None:
        # datamodule.prepare_data()
        datamodule.setup()
        return

    # display dataset
    # datamodule.prepare_data()
    datamodule.setup()
    if config.verbose:
        rich.print(datamodule.dataset)
        pprint_batch(next(iter(datamodule.train_dataloader())), "training batch")
        datamodule.display_samples(n_samples=1)

    # Init Lightning Module
    log.info(f"Instantiating Module <{config.model._target_}>")
    model: LightningModule = instantiate(config.model, _recursive_=False)

    # Init Lightning callbacks, attach the datamodule and the model tot he IndexOpenQa callback
    callbacks: List[Callback] = []
    if config.get("callbacks", None):
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callback = instantiate(cb_conf)
                if isinstance(callback, IndexOpenQaCallback):
                    log.info(
                        f"Attaching datamodule <{type(datamodule).__name__}> "
                        f"to {type(callback).__name__}"
                    )
                    callback.attach(datamodule)

                callbacks.append(callback)

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

    # Log config to all lightning loggers
    log.info("Logging hyperparameters.")
    train_utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Training...
    patch_signal_connector(trainer)
    mapping_freq = config.datamodule.get("mapping_freq", None)
    if mapping_freq is not None and mapping_freq > 0:
        log.info(
            f"Starting training with dataset updates "
            f"(max_epochs={trainer.max_epochs}, mapping_freq={mapping_freq}).."
        )
        train_with_dataset_updates(datamodule, model, trainer, mapping_freq)
    else:
        log.info(f"Starting training (max_epochs={trainer.max_epochs})..")
        trainer.fit(model=model, datamodule=datamodule)

    # Evaluate Module on test set after training
    if not config.trainer.get("fast_dev_run"):
        log.info("Starting testing..")
        trainer.test(dataloaders=datamodule.test_dataloader())

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
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]


def patch_signal_connector(trainer: Trainer):
    """
    Avoid using `signal` in `trainer.SignalConnector` if Lightning
    is not running in the main thread. See:
    https://github.com/PyTorchLightning/pytorch-lightning/issues/9590#issuecomment-992038707
    """
    if threading.current_thread() is threading.main_thread():
        return

    warnings.warn(
        "Lightning is not running in the main thread. "
        "Patching `trainer.SignalConnector` to avoid using `signal`."
    )

    def _no_signal_teardown(self):
        self._original_handlers = {}

    trainer.signal_connector._is_on_windows = lambda *_: True
    trainer.signal_connector.teardown = partial(_no_signal_teardown, trainer.signal_connector)


def train_with_dataset_updates(datamodule, model, trainer, update_freq: int):
    """Fit the model to the dataset, updating the dataset every `update_freq` epochs."""
    max_epochs = trainer.max_epochs
    trainer.fit_loop.max_epochs = min(update_freq, max_epochs)
    while trainer.current_epoch < max_epochs:
        try:
            trainer.fit(
                model=model,
                train_dataloader=datamodule.train_dataloader(),
                val_dataloaders=datamodule.val_dataloader(),
            )

            log.info(f"Epoch {trainer.current_epoch} completed.")
            if trainer.current_epoch >= max_epochs:
                break

            log.info("Updating dataset...")
            start_time = time.time()
            datamodule.update_dataset(model=model, trainer=trainer, keep_in_memory=True)
            log.info(f"Dataset updated in {time.time() - start_time:.2f}s")

            trainer.fit_loop.max_epochs += update_freq
            trainer.num_sanity_val_steps = 0
        except KeyboardInterrupt:
            log.info("Training interrupted.")
            break
