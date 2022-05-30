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
from fz_openqa.utils.elasticsearch import ElasticSearchInstance
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
    sharing_strategy = config.get("base.sharing_strategy", "file_system")
    log.info(f"Using {sharing_strategy} sharing strategy")
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

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

    with ElasticSearchInstance(
        disable=not config.get("spawn_es", False), stdout=open("es.stdout.log", "w")
    ):

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
            for cid, cb_conf in config["callbacks"].items():
                if cb_conf is not None and "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callback = instantiate(cb_conf)
                    callbacks.append(callback)
                else:
                    log.warning(f"Skipping callback {cid}: <{cb_conf}>")

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
        setup_model = instantiate_setup_model(
            config.get("setup_with_model", None), main_model=model
        )
        # datamodule.prepare_data()
        datamodule.setup(trainer=trainer, model=setup_model)
        if config.verbose:
            rich.print(datamodule.dataset)
            pprint_batch(next(iter(datamodule.train_dataloader())), "training batch")
            datamodule.display_samples(n_samples=1)

    # Log config to all lightning loggers
    train_utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # eval on init
    trainer.validate(model=model, dataloaders=datamodule.val_dataloader())

    # Training...
    patch_signal_connector(trainer)
    dataset_update = config.datamodule.get("dataset_update", None)
    if dataset_update is not None:
        dataset_update_freq = dataset_update["freq"]
        log.info(
            f"Starting training with dataset updates "
            f"(max_steps={trainer.max_steps}, freq={dataset_update_freq} steps).."
        )
        train_with_dataset_updates(
            datamodule,
            model=model,
            trainer=trainer,
            update_freq=dataset_update_freq,
            test_every_update=dataset_update.get("test_every_update", False),
            reset_optimizer=dataset_update.get("reset_optimizer", True),
            reset_parameters=dataset_update.get("reset_parameters", False),
            spawn_es=config.get("spawn_es", False),
            **dataset_update.get("builder_args", {}),
        )
    else:
        log.info(f"Starting training (max_steps={trainer.max_steps})..")
        trainer.fit(model=model, datamodule=datamodule)

    # Evaluate Module on test set after training
    if not config.trainer.get("fast_dev_run"):
        log.info("Starting testing..")
        trainer.test(model=model, dataloaders=datamodule.test_dataloader())

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
    reset_parameters: bool = False,
    test_every_update: bool = True,
    load_best_model: bool = False,
    index_on_first_step: bool = False,
    keep_in_memory: bool = False,
    spawn_es: bool = False,
    **kwargs,
) -> LightningModule:
    """Fit the model to the dataset, updating the dataset every `update_freq` epochs."""
    max_steps = trainer.max_steps
    trainer.fit_loop.max_steps = min(update_freq, max_steps)
    dataset_iter = 0
    trainer.logger.log_metrics({"dataset_update/step": dataset_iter}, step=trainer.global_step)
    while trainer.global_step < max_steps:

        # update the dataset
        try:
            if trainer.global_step > 0 or index_on_first_step:
                dataset_iter += 1
                update_dataset(
                    datamodule,
                    model=model,
                    trainer=trainer,
                    keep_in_memory=keep_in_memory,
                    dataset_iter=dataset_iter,
                    spawn_es=spawn_es,
                    **kwargs,
                )
                if test_every_update:
                    log.info(f"Starting testing (update={dataset_iter})..")
                    trainer.test(model=model, dataloaders=datamodule.test_dataloader())
        except Exception:
            log.exception("Dataset update interrupted.")
            break

        # fit the model for `update_freq` epochs
        try:
            log.info(
                f"Starting training for "
                f"{trainer.fit_loop.max_steps - trainer.global_step} steps"
                f" (update={dataset_iter}).."
            )
            trainer.fit(
                model=model,
                train_dataloaders=datamodule.train_dataloader(),
                val_dataloaders=datamodule.val_dataloader(),
            )
            log.info(f"Step {trainer.global_step} completed, epoch {trainer.current_epoch}.")

            if trainer.max_steps > max_steps:
                break
            if trainer.state.status == TrainerStatus.INTERRUPTED:
                log.info(
                    f"Training interrupted. " f"Steps remaining: {max_steps - trainer.global_step}"
                )
                break

            # update trainer parameters
            trainer.fit_loop.max_steps += update_freq
            trainer.num_sanity_val_steps = 0

            # get optimizer state and store it into the model, so it can be
            # set in the beginning of `trainer.fit()`
            if not reset_optimizer:
                set_model_opt_states(model)
            if reset_parameters:
                try:
                    param_names = list(model._params.parameters.keys())
                    log.info(f"Resetting model parameter schedules {param_names}")
                    model._params.reset()
                except Exception as exc:
                    log.error(f"Couldn't reset parameters: {exc}")
        except KeyboardInterrupt:
            log.info("Training interrupted.")
            break

    # load the best model and return it
    log.info("Training completed. Loading best model and re-indexing the dataset")
    if load_best_model and trainer.checkpoint_callback.last_model_path is not None:
        model = model.load_from_checkpoint(trainer.checkpoint_callback.last_model_path)
    else:
        log.info("No checkpoint found. Using the last model.")

    update_dataset(
        datamodule, model=model, trainer=trainer, splits=[Split.TEST], spawn_es=spawn_es, **kwargs
    )

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
    spawn_es: bool = False,
    **kwargs,
):
    log.info("Updating dataset...")
    with ElasticSearchInstance(disable=not spawn_es, stdout=open("es.stdout.log", "w")):
        start_time = time.time()
        datamodule.update_dataset(
            model=model, trainer=trainer, keep_in_memory=keep_in_memory, **kwargs
        )
        # datamodule.display_samples(n_samples=1)
        elapsed_time = time.time() - start_time
        log.info(f"Dataset updated in {elapsed_time:.2f}s")
        trainer.logger.log_metrics(
            {"dataset_update/step": dataset_iter, "dataset_update/time": elapsed_time},
            step=trainer.global_step,
        )
