import logging
import os
import threading
import time
import warnings
from functools import partial
from typing import List
from typing import Optional

import datasets
import pytorch_lightning as pl
import rich
import torch
from datasets import Split
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loggers import LightningLoggerBase
from torch.optim.lr_scheduler import LambdaLR

from fz_openqa.callbacks.index_openqa import IndexOpenQaCallback
from fz_openqa.datamodules import DataModule
from fz_openqa.modeling import Model
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
    #    datamodule.display_samples(n_samples=1)

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
    dataset_update = config.datamodule.get("dataset_update", None)
    if dataset_update is not None:
        dataset_update_freq = dataset_update["freq"]
        log.info(
            f"Starting training with dataset updates "
            f"(max_epochs={trainer.max_epochs}, dataset_update_freq={dataset_update_freq}).."
        )
        train_with_dataset_updates(
            datamodule,
            model=model,
            trainer=trainer,
            update_freq=dataset_update_freq,
            reset_optimizer=dataset_update.get("reset_optimizer", True),
            **dataset_update.get("builder_args", {}),
        )
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


def train_with_dataset_updates(
    datamodule: DataModule,
    *,
    model: Model,
    trainer: Trainer,
    update_freq: int,
    reset_optimizer: bool = False,
    **kwargs,
) -> LightningModule:
    """Fit the model to the dataset, updating the dataset every `update_freq` epochs."""
    max_epochs = trainer.max_epochs
    trainer.fit_loop.max_epochs = min(update_freq, max_epochs)
    while trainer.current_epoch < max_epochs:

        # update the dataset
        try:
            if trainer.current_epoch > 0:
                update_dataset(
                    datamodule, model=model, trainer=trainer, keep_in_memory=True, **kwargs
                )
        except Exception:
            log.exception("Dataset update interrupted.")
            break

        # fit the model for `update_freq` epochs
        try:
            trainer.fit(
                model=model,
                train_dataloader=datamodule.train_dataloader(),
                val_dataloaders=datamodule.val_dataloader(),
            )

            log.info(f"Epoch {trainer.current_epoch} completed.")
            if trainer.current_epoch >= max_epochs:
                break
            elif trainer.current_epoch < trainer.fit_loop.max_epochs - 1:
                log.info(
                    f"Training interrupted. "
                    f"Epochs remaining: {trainer.fit_loop.max_epochs - trainer.current_epoch}"
                )
                break

            # update trainer parameters
            trainer.fit_loop.max_epochs += update_freq
            trainer.num_sanity_val_steps = 0

            # get optimizer state and store it into the model, so it can be
            # set in the beginning of `trainer.fit()`
            if not reset_optimizer:
                set_model_opt_states(model)
        except KeyboardInterrupt:
            log.info("Training interrupted.")
            break

    # load the best model and return it
    log.info("Training completed. Loading best model and re-indexing the dataset")
    if trainer.checkpoint_callback.last_model_path is not None:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.last_model_path)
    else:
        log.info("No checkpoint found. Using the last model.")

    update_dataset(datamodule, model=model, trainer=trainer, splits=[Split.TEST], **kwargs)

    return model


def set_model_opt_states(model):
    optimizer = model.optimizers()
    if isinstance(optimizer, LightningOptimizer):
        optimizer = optimizer.optimizer
    scheduler = model.lr_schedulers()
    model.opt_states = {"optimizer": optimizer.state_dict(), "lr_scheduler": scheduler.state_dict()}


def update_dataset(
    datamodule: DataModule,
    *,
    model: pl.LightningModule,
    trainer: Trainer,
    keep_in_memory=True,
    **kwargs,
):
    log.info("Updating dataset...")
    start_time = time.time()
    datamodule.update_dataset(model=model, trainer=trainer, keep_in_memory=keep_in_memory, **kwargs)
    log.info(f"Dataset updated in {time.time() - start_time:.2f}s")
