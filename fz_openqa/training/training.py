from __future__ import annotations

import logging
import os
import threading
import time
import warnings
from functools import partial
from typing import List
from typing import Optional

import datasets
import jsondiff
import pytorch_lightning as pl
import rich
import torch
from datasets import Split
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.trainer.states import TrainerStatus

from fz_openqa.datamodules import DataModule
from fz_openqa.inference.checkpoint import CheckpointLoader
from fz_openqa.modeling import Model
from fz_openqa.utils import train_utils
from fz_openqa.utils.pretty import get_separator
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

    # load checkpoint manager
    checkpoint_manager = load_checkpoint(
        config.get("checkpoint", None),
        override_config=DictConfig({"sys": config.sys}),
        ref_config=config,
    )
    if checkpoint_manager is not None:
        override_config(config, checkpoint_manager.config, config.get("config_overrides", []))

    # log paths
    log.info(f"work_dir={config.sys.work_dir}")
    log.info(f"cache_dir={os.path.abspath(config.sys.cache_dir)}")
    log.info(f"Experiment working directory: {os.getcwd()}")

    # # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.base.seed, workers=True)

    # only preprocess the data if there is no trainer
    if config.get("trainer", None) is None:
        log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
        datamodule: DataModule = instantiate(config.datamodule)
        # datamodule.prepare_data()
        datamodule.setup()
        return

    # Init Lightning Module
    model = instantiate_model(
        config,
        checkpoint_manager=checkpoint_manager,
        restore_from_checkpoint=config.get("restore_from_checkpoint", False),
    )

    # Init Lightning callbacks, attach the datamodule and the model tot he IndexOpenQa callback
    callbacks: List[Callback] = []
    if config.get("callbacks", None):
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callback = instantiate(cb_conf)
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
    # todo: resume trainer from checkpoint with state (require updating Lightning)
    trainer: Trainer = instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # instantiate the datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: DataModule = instantiate(config.datamodule)
    setup_model = instantiate_setup_model(config.get("setup_with_model", None), main_model=model)
    # datamodule.prepare_data()
    datamodule.setup(trainer=trainer, model=setup_model)
    if config.verbose:
        rich.print(datamodule.dataset)
        pprint_batch(next(iter(datamodule.train_dataloader())), "training batch")
        datamodule.display_samples(n_samples=1)

    # potentially kill elastic search
    if config.get("kill_es", False):
        log.info("Killing elasticsearch.")
        os.system("pkill -f elasticsearch")

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
            index_first_epoch=dataset_update.get("index_first_epoch", False),
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


def override_config(config: DictConfig, from_config: DictConfig, overrides: List):
    for key_path in overrides:
        ref = config
        ckpt = from_config
        *key_path, final_key = key_path.split(".")
        for key in key_path:
            ref = ref[key]
            ckpt = ckpt[key]
        ref[final_key] = ckpt[final_key]


def instantiate_model(
    config: DictConfig,
    *,
    checkpoint_manager: Optional[CheckpointLoader] = None,
    restore_from_checkpoint=False,
):
    """Instantiate the model from the config or load from checkpoint."""
    if checkpoint_manager is None:
        restore_from_checkpoint = False
    if not restore_from_checkpoint:
        log.info(f"Instantiating Module <{config.model._target_}>")
        model: Model = instantiate(config.model, _recursive_=False)
    else:
        log.info(f"Loading Module <{config.model._target_}> from checkpoint")
        model: Model = checkpoint_manager.load_model(last=config.get("checkpoint_type", "last"))
    return model


def load_checkpoint(
    checkpoint_path: Optional[str],
    *,
    override_config: Optional[DictConfig],
    ref_config: Optional[DictConfig],
) -> Optional[CheckpointLoader]:
    """Load a CheckpointLoader from a checkpoint path and print the
    difference between the `checkpoint.config` and the `config`."""
    if checkpoint_path is not None:
        checkpoint = CheckpointLoader(checkpoint_path, override=override_config)
        # todo: move to original directory
        # todo: os.chdir(checkpoint.config.sys.workdir)
        # todo: config.sys.workdir = checkpoint.config.sys.workdir
        if ref_config is not None:
            rich.print(get_separator())
            rich.print(f"Loading checkpoint from {checkpoint_path}. Config diff:")
            rich.print(
                jsondiff.diff(
                    OmegaConf.to_container(checkpoint.config, resolve=False),
                    OmegaConf.to_container(ref_config, resolve=False),
                    syntax="symmetric",
                )
            )
            rich.print(get_separator())
        return checkpoint
    else:
        return None


def instantiate_setup_model(setup_with_model: DictConfig, *, main_model: Model) -> Optional[Model]:
    """
    Instantiate the model used to setup the dataset.
    if `setup_with_model` is a string, it will be interpreted as the path to a checkpoint.
    if `setup_with_model` is `True`, the main model will be used.
    """

    if isinstance(setup_with_model, str):
        log.info(f"Setup model: Instantiating from path <{setup_with_model}>")
        return CheckpointLoader(setup_with_model).load_model(checkpoint_type="best")
    elif isinstance(setup_with_model, bool) and setup_with_model:
        log.info(f"Setup model: Using main model <{type(main_model)}>")
        return main_model
    else:
        return None


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
    index_first_epoch: bool = False,
    test_every_update: bool = True,
    **kwargs,
) -> LightningModule:
    """Fit the model to the dataset, updating the dataset every `update_freq` epochs."""
    max_epochs = trainer.max_epochs
    trainer.fit_loop.max_epochs = min(update_freq, max_epochs)
    dataset_iter = 0
    trainer.logger.log_metrics({"dataset_update/step": dataset_iter}, step=trainer.global_step)
    while trainer.current_epoch < max_epochs:

        # update the dataset
        try:
            if index_first_epoch or trainer.current_epoch > 0:
                dataset_iter += 1
                update_dataset(
                    datamodule,
                    model=model,
                    trainer=trainer,
                    keep_in_memory=True,
                    dataset_iter=dataset_iter,
                    **kwargs,
                )
                if test_every_update:
                    log.info(f"Starting testing (update={dataset_iter})..")
                    trainer.test(dataloaders=datamodule.test_dataloader())
        except Exception:
            log.exception("Dataset update interrupted.")
            break

        # fit the model for `update_freq` epochs
        try:
            log.info(
                f"Starting training for "
                f"{trainer.fit_loop.max_epochs - trainer.current_epoch} epochs"
                f" (update={dataset_iter}).."
            )
            trainer.fit(
                model=model,
                train_dataloader=datamodule.train_dataloader(),
                val_dataloaders=datamodule.val_dataloader(),
            )
            log.info(f"Epoch {trainer.current_epoch} completed.")

            # increment the epoch counter by one, seems to be missing in the original code
            trainer.fit_loop.current_epoch += 1
            if trainer.current_epoch > max_epochs:
                break
            if trainer.state.status == TrainerStatus.INTERRUPTED:
                log.info(
                    f"Training interrupted. "
                    f"Epochs remaining: {max_epochs - trainer.current_epoch}"
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
    dataset_iter: int = 0,
    **kwargs,
):
    log.info("Updating dataset...")
    start_time = time.time()
    datamodule.update_dataset(model=model, trainer=trainer, keep_in_memory=keep_in_memory, **kwargs)
    # datamodule.display_samples(n_samples=1)
    elapsed_time = time.time() - start_time
    log.info(f"Dataset updated in {elapsed_time:.2f}s")
    trainer.logger.log_metrics(
        {"dataset_update/step": dataset_iter, "dataset_update/time": elapsed_time},
        step=trainer.global_step,
    )
