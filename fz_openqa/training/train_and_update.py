from __future__ import annotations

import math
import time
from typing import Optional

import pytorch_lightning as pl
from datasets import Split
from loguru import logger as log
from pydantic import BaseModel
from pydantic import PositiveInt
from pydantic import validator
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.trainer.states import TrainerStatus

from fz_openqa.datamodules import DataModule
from fz_openqa.modeling import Model
from fz_openqa.utils.elasticsearch import ElasticSearchInstance


class DatasetUpdateConfig(BaseModel):
    freq: Optional[PositiveInt]
    reset_optimizer: bool = True
    reset_parameter_scheduler: bool = False
    reset_parameters: bool = False
    test_every_update: bool = False
    index_on_first_step: bool = False
    spawn_es: bool = True


def train_with_dataset_updates(
    datamodule: DataModule,
    *,
    model: Model,
    trainer: Trainer,
    config: DatasetUpdateConfig,
    load_best_model: bool = False,
    **kwargs,
) -> LightningModule:
    """Fit the model to the dataset, updating the dataset every `update_freq` epochs."""
    log.info(
        f"Starting training with dataset updates "
        f"(max_steps={trainer.max_steps}, update_freq={config.freq}).."
    )

    max_steps = trainer.max_steps
    freq = max_steps if config.freq is None else min(config.freq, max_steps)
    n_iters = math.ceil(max_steps / freq)
    trainer.fit_loop.max_steps = freq
    dataset_iter = 0
    if trainer.logger is not None:
        trainer.logger.log_metrics({"dataset_update/step": dataset_iter}, step=trainer.global_step)
    while trainer.global_step < max_steps - 1:

        # update the dataset
        try:
            if trainer.global_step > 0 or config.index_on_first_step:
                dataset_iter += 1
                update_dataset(
                    datamodule,
                    model=model,
                    trainer=trainer,
                    dataset_iter=dataset_iter,
                    **kwargs,
                )
                if config.test_every_update:
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
                f" (update={dataset_iter}/{n_iters}).."
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
            trainer.fit_loop.max_steps += freq
            trainer.num_sanity_val_steps = 0

            # get optimizer state and store it into the model, so it can be
            # set in the beginning of `trainer.fit()`
            if not config.reset_optimizer:
                set_model_opt_states(model)

            if config.reset_parameter_scheduler:
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
    try:
        if load_best_model and trainer.checkpoint_callback.last_model_path is not None:
            model = model.load_from_checkpoint(trainer.checkpoint_callback.last_model_path)
        else:
            log.info("No checkpoint found. Using the last model.")
    except Exception as exc:
        log.error(f"Couldn't load the best model: {exc}")

    # re-index the dataset
    if config.freq is not None:
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
    dataset_iter: int = 0,
    spawn_es: bool = False,
    **kwargs,
):
    log.info("Updating dataset...")
    with ElasticSearchInstance(disable=not spawn_es, stdout=open("es.stdout.log", "w")):
        start_time = time.time()
        datamodule.update_dataset(model=model, trainer=trainer, **kwargs)
        # datamodule.display_samples(n_samples=1)
        elapsed_time = time.time() - start_time
        log.info(f"Dataset updated in {elapsed_time:.2f}s")
        trainer.logger.log_metrics(
            {"dataset_update/step": dataset_iter, "dataset_update/time": elapsed_time},
            step=trainer.global_step,
        )
