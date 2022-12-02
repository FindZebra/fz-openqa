from __future__ import annotations

import logging
import os
import threading
import warnings
from functools import partial
from typing import List
from typing import Optional

import datasets
import jsondiff
import rich
import torch
from hydra.utils import instantiate
from loguru import logger as log
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from warp_pipes import get_console_separator

from fz_openqa.datamodules import DataModule
from fz_openqa.inference.checkpoint import CheckpointLoader
from fz_openqa.modeling import Model
from fz_openqa.training.train_and_update import train_with_dataset_updates
from fz_openqa.utils import train_utils
from fz_openqa.utils.elasticsearch import ElasticSearchInstance
from fz_openqa.utils.train_utils import setup_safe_env


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
    logging.getLogger("elasticsearch").setLevel(logging.ERROR)
    datasets.logging.set_verbosity(datasets.logging.CRITICAL)

    # set default cache dirs
    os.environ["HF_DATASETS_CACHE"] = str(config.sys.cache_dir)
    os.environ["HF_TRANSFORMERS_CACHE"] = str(config.sys.cache_dir)

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
        disable=not config.datamodule.dataset_update.spawn_es,
        stdout=open("es.stdout.log", "w"),
    ):

        # only preprocess the data if there is no trainer
        if "null" in config.trainer._target_:
            log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
            datamodule: DataModule = instantiate(config.datamodule)
            # datamodule.prepare_data()
            datamodule.setup()
            datamodule.display_samples(n_samples=10)
            rich.print(datamodule.dataset)
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
            config.get("setup_with_model", None),
            main_model=model,
            override_config=DictConfig({"sys": config.sys}),
            ref_config=config,
        )
        # datamodule.prepare_data()
        datamodule.setup(trainer=trainer, model=setup_model)
        if config.verbose:
            rich.print(datamodule.dataset)
            datamodule.display_samples(n_samples=1)

    # Log config to all lightning loggers
    train_utils.log_hyperparameters(
        config=config,
        model=model,
        trainer=trainer,
    )

    # eval on init
    trainer.validate(model=model, dataloaders=datamodule.val_dataloader())

    # Training...
    patch_signal_connector(trainer)
    dataset_update = instantiate(config.datamodule.dataset_update)

    train_with_dataset_updates(
        datamodule,
        model=model,
        trainer=trainer,
        config=dataset_update,
        load_best_model=config.get("test_with_best_model", True),
    )

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
        model: Model = checkpoint_manager.load_model(
            checkpoint_type=config.get("checkpoint_type", "last")
        )
    return model


def load_checkpoint(
    checkpoint_path: Optional[str],
    *,
    override_config: Optional[DictConfig],
    ref_config: Optional[DictConfig],
    silent: bool = False,
) -> Optional[CheckpointLoader]:
    """Load a CheckpointLoader from a checkpoint path and print the
    difference between the `checkpoint.config` and the `config`."""
    if checkpoint_path is not None:
        checkpoint = CheckpointLoader(
            checkpoint_path, override=override_config, cache_dir=ref_config.sys.cache_dir
        )
        # todo: move to original directory
        # todo: os.chdir(checkpoint.config.sys.workdir)
        # todo: config.sys.workdir = checkpoint.config.sys.workdir
        if ref_config is not None and not silent:
            rich.print(get_console_separator())
            rich.print(f"Loading checkpoint from {checkpoint_path}. Config diff:")
            rich.print(
                jsondiff.diff(
                    OmegaConf.to_container(checkpoint.config, resolve=False),
                    OmegaConf.to_container(ref_config, resolve=False),
                    syntax="symmetric",
                )
            )
            rich.print(get_console_separator())
        return checkpoint
    else:
        return None


def instantiate_setup_model(
    setup_with_model: DictConfig,
    *,
    main_model: Model,
    override_config: Optional[DictConfig],
    ref_config: Optional[DictConfig],
) -> Optional[Model]:
    """
    Instantiate the model used to setup the dataset.
    if `setup_with_model` is a string, it will be interpreted as the path to a checkpoint.
    if `setup_with_model` is `True`, the main model will be used.
    """

    if isinstance(setup_with_model, str):
        log.info(f"Setup model: Instantiating from path <{setup_with_model}>")
        checkpoint = CheckpointLoader(
            setup_with_model, override=override_config, cache_dir=ref_config.sys.cache_dir
        )
        return checkpoint.load_model(checkpoint_type=ref_config.get("checkpoint_type", "last"))
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
