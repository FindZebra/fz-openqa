import json
import os
from typing import List
from typing import Optional

import datasets
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from rich.progress import track

from fz_openqa.datamodules import DataModule
from fz_openqa.datamodules.pipes import DropKeys
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import torch
from fz_openqa.datamodules.pipes import UpdateWith
from fz_openqa.datamodules.pipes.nesting import infer_batch_size
from fz_openqa.datamodules.pipes.update import UpdateKeys
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
        datasets.logging.set_verbosity(datasets.logging.ERROR)
        # datasets.disable_progress_bar()

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
        datamodule.prepare_data()
        datamodule.setup()

        corpus_name = str(config.datamodule.builder.corpus_builder._id)
        cls_name = str(config.datamodule.relevance_classifier._id)
        trial_path = os.getcwd()

        dataloaders = {
            "train": datamodule.train_dataloader(),
            "val": datamodule.val_dataloader(),
            "test": datamodule.test_dataloader(),
        }

        drop_fields = DropKeys(
            keys=[
                "answer.attention_mask",
                "answer.input_ids",
                "document.attention_mask",
                "document.input_ids",
                "question.attention_mask",
                "question.input_ids",
            ]
        )
        update_fields = UpdateKeys(lambda v: isinstance(v, torch.Tensor))
        pipe = UpdateWith(Sequential(drop_fields, update_fields))

        for split, dataloader in dataloaders.items():
            with open(
                os.path.join(trial_path, f"{corpus_name}-{cls_name}-{split}.jsonl"), mode="w"
            ) as fp:
                for batch in track(
                    dataloader, description=f"Iterating through the {split} dataset..."
                ):

                    batch = pipe(batch)

                    for i in range(infer_batch_size(batch)):
                        row = Pipe.get_eg(batch, idx=i)
                        fp.write(f"{json.dumps(row)}\n")
        return

    # Init Lightning Module
    log.info(f"Instantiating Module <{config.model._target_}>")
    model: LightningModule = instantiate(config.model, _recursive_=False)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if config.get("callbacks", None):
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(instantiate(cb_conf))

    # Init Lightning loggers # todo: check this
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
    log.info("Starting training..")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate Module on test set after training
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
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
