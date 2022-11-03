from __future__ import annotations

from copy import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import rich
import torch
from datasets import Split
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerFast
from warp_pipes import Batch

from fz_openqa.modeling.modules.base import Module
from fz_openqa.modeling.parameters import Parameters
from fz_openqa.utils import maybe_instantiate
from fz_openqa.utils.functional import is_loggable
from fz_openqa.utils.functional import only_trainable


class Model(LightningModule):
    """
    This class implements the basics of evaluation, logging and inference using
    pytorch lightning mechanics. It contains a model: `nn.Module`.

    ## Main components
    This class contains 2 main components:
    * self.backbone: the pretrained masked language model
    * self.module: define the actual computation of the model
    * self.evaluator: handles computing the loss within the module and evaluate the metrics

    ## Pipeline
    The main data processing flow can be described as follows:

        1.     batch = next(iter(dataloader))          (device=k)
                            |
            [   _step(batch): evaluator.step   ]    (processing on device k)
                            v
        2.             pre_output                      (device=k)
                            |
                  [ gather (lightning) ]               (move data to device 0)
                            v
        3.              pre_output                     (device=0)
                            |
    [ _step_end(pre_output): evaluator.step_end + log_data ]
                            v
        4.              output                         (device=0)


    ## Metrics:
    The evaluator keeps track of the metrics using `torchmetrics`.
    The metrics are updated at each `_step_end` (e.g. keeping track of
    the true positives and false negatives).
    The metrics are computed for the whole epoch in `_epoch_end`.
    """

    tracked_metrics: Optional[Dict[str, Optional[Tensor]]] = None

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast | DictConfig,
        backbone: PreTrainedModel | DictConfig,
        module: DictConfig | Module,
        monitor_metric: Optional[str],
        optimizer: torch.optim.Optimizer | DictConfig,
        lr_scheduler: torch.optim.lr_scheduler.ChainedScheduler | DictConfig,
        parameters: Optional[Parameters | Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        # the following keys must be registered in the .hparams, so the objects
        # can be instantiated in `self.configure_optimizers()`
        for key in ["optimizer", "lr_scheduler", "monitor_metric"]:
            if key not in self.hparams:
                raise TypeError(
                    f"{key} must be registered with `save_hyperparameters()`. "
                    f"Found {self.hparams.keys()}"
                )
        for key in ["optimizer", "lr_scheduler"]:
            if not isinstance(self.hparams[key], (dict, DictConfig)):
                raise TypeError(
                    f"Config for {key} must be provided, found {type(self.hparams[key])}"
                )

        # store the state of the optimizer
        self.opt_states: Optional[Dict] = None

        # instantiate the model
        self.module: Optional[Module] = maybe_instantiate(
            module,
            backbone=backbone,
            tokenizer=tokenizer,
            _recursive_=False,
        )

        # parameters
        if parameters is None:
            self._params = None
        elif isinstance(parameters, (dict, DictConfig)):
            if "_target_" in parameters.keys():
                self._params = maybe_instantiate(parameters)
            else:
                self._params = Parameters(**parameters)
        else:
            self._params = parameters

        logger.info(f"Parameters: {self._params}")

    @property
    def params(self) -> Dict[str, float]:
        if self._params is None:
            return {}
        return self._params()

    def forward(self, batch: Batch, **kwargs) -> Batch:
        return self.module.forward(batch, **kwargs, **self.params)

    def predict(self, batch: Batch, **kwargs) -> Batch:
        return self.module.predict(batch, **kwargs, **self.params)

    def evaluate(self, batch: Batch, *, split: Split = Split.TRAIN, **kwargs) -> Batch:
        output = self._step(batch, split=split, **kwargs)
        output_ = self._step_end(output, split=split, **kwargs)
        return {**output, **output_}

    def _step(
        self,
        batch: Batch,
        batch_idx: int = None,
        dataloader_idx: Optional[int] = None,
        *,
        split: Split,
        **kwargs,
    ) -> Batch:
        """
        Perform the model forward pass and compute the loss or pre loss terms.
        !! This step is performed separately on each device. !!
        """
        return self.module.step(batch, **kwargs, **self.params)

    def _step_end(self, pre_output: Batch, *, split: Split, log_data=True) -> Tensor | Batch:
        """
        Call the `evaluator.forward_end` method (finalize the loss computation
        and update the metrics) using the `pre_output` data gathered from
        all devices.

        !! This step is performed on device 0 !!
        """
        output = self.module.step_end(pre_output, split)

        # track metrics
        if self.tracked_metrics is not None:
            for key in self.tracked_metrics.keys():
                x = self.tracked_metrics[key]
                y = output[key].detach().cpu()
                if x is None:
                    self.tracked_metrics[key] = y
                else:
                    self.tracked_metrics[key] = torch.cat([x, y], dim=0)

        if log_data:
            if self._params is not None:
                output = copy(output)
                output.update({f"parameters/{k}": v for k, v in self.params.items()})
            # potentially log the loss and
            # other metrics that are computed on each step
            on_step = str(split) == (Split.TRAIN)
            self.log_data(output, prefix=str(split), on_step=on_step, on_epoch=not on_step)

        return output

    def _epoch_end(self, outputs: List[Any], *, split: Split, log_data=True) -> Batch:
        """
        1. Compute the metrics for the whole epoch using `evaluator.compute_metrics`
        2. Log the metrics for the whole epoch
        """
        assert self.module is not None
        metrics = self.module.compute_metrics(split=split)
        if log_data:
            self.log_data(metrics, prefix=str(split))
        self.module.reset_metrics(split=split)
        return metrics

    @torch.no_grad()
    def log_data(
        self,
        data: Batch,
        prefix: Optional[str] = None,
        on_step=False,
        on_epoch=True,
        sync_dist=True,
    ):
        """
        Log all data from the input Batch. Only tensors with one elements are logged.
        Each key is formatted as: `prefix/key` where prefix is usually the `Split`.
        """
        for k, v in data.items():
            key = "/".join(u for u in (prefix, self.module.task_id, k) if u is not None)
            if is_loggable(v):
                self.log(
                    key,
                    v,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=key in self.module.pbar_metrics,
                    sync_dist=sync_dist,
                )

    def configure_optimizers(self):
        """
        Configure the optimizer and the learning rate scheduler.
        """

        # optimizer and scheduler
        optimizer_cfg = copy(self.hparams.optimizer)
        lr_scheduler_cfg = copy(self.hparams.lr_scheduler)
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight", "BayesianLinear"]
        weight_decay = optimizer_cfg.get("weight_decay", 0.0)
        optimizer_grouped_parameters = [
            {
                "params": list(
                    only_trainable(
                        [
                            p
                            for n, p in self.named_parameters()
                            if not any(nd in n for nd in no_decay)
                        ]
                    )
                ),
                "weight_decay": weight_decay,
            },
            {
                "params": list(
                    only_trainable(
                        [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]
                    )
                ),
                "weight_decay": 0.0,
            },
        ]

        for group in optimizer_grouped_parameters:
            logger.info(
                f"Optimizer group: "
                f"n_params={len(group['params'])}, "
                f"weight_decay={group['weight_decay']}"
            )

        # define the optimizer using the above groups
        optimizer = maybe_instantiate(
            optimizer_cfg, _args_=[optimizer_grouped_parameters], _convert_="all"
        )

        # defile the learning rate scheduler
        lr_scheduler = maybe_instantiate(lr_scheduler_cfg, _args_=[optimizer], _convert_="all")

        # if an optimizer state is available, set it
        # this is a trick to avoid resetting the state between multiple `trainer.fit()` steps
        if self.opt_states is not None:
            opt_state = self.opt_states.pop("optimizer", None)
            scheduler_state = self.opt_states.pop("lr_scheduler", None)
            if opt_state is not None:
                logger.warning("Setting Optimizer state!")
                optimizer.load_state_dict(opt_state)
            if scheduler_state is not None and lr_scheduler is not None:
                logger.warning("Setting Scheduler state!")
                lr_scheduler.load_state_dict(scheduler_state)
            self.opt_states = None

        output = {
            "optimizer": optimizer,
            "lr_scheduler": {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": self.hparams.monitor_metric,
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            },
        }
        return output

    def on_before_zero_grad(self, *args, **kwargs):
        if self._params is not None:
            self._params.step()

    def training_step_end(self, batch: Batch, **kwargs) -> Batch:
        return self._step_end(batch, split=Split.TRAIN)

    def validation_step_end(self, batch: Batch, **kwargs) -> Batch:
        return self._step_end(batch, split=Split.VALIDATION)

    def test_step_end(self, batch: Batch, **kwargs) -> Batch:
        return self._step_end(batch, split=Split.TEST)

    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Batch:
        return self._step(batch, batch_idx, dataloader_idx, split=Split.TRAIN)

    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Batch:
        return self._step(batch, batch_idx, dataloader_idx, split=Split.VALIDATION)

    def test_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Batch:
        return self._step(batch, batch_idx, dataloader_idx, split=Split.VALIDATION)

    def training_epoch_end(self, outputs: List[Any]):
        self._epoch_end(outputs, split=Split.TRAIN)

    def validation_epoch_end(self, outputs: List[Any]):
        self._epoch_end(outputs, split=Split.VALIDATION)

    def test_epoch_end(self, outputs: List[Any]):
        self._epoch_end(outputs, split=Split.TEST)

    def check_input_features(self, batch):
        for f in self._required_feature_names:
            assert f in batch.keys(), f"The feature {f} is required."
