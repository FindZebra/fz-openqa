from typing import Any
from typing import List
from typing import Optional
from typing import Union

from datasets import Split
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from transformers import AdamW
from transformers import BertPreTrainedModel
from transformers import PreTrainedTokenizerFast

from fz_openqa.modeling.modules.base import Module
from fz_openqa.utils import maybe_instantiate
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import is_loggable
from fz_openqa.utils.functional import only_trainable


class Model(LightningModule):
    """
    This class implements the basics of evaluation, logging and inference using
    pytorch lightning mechanics. It contains a model: `nn.Module`.

    ## Main components
    This class contains 2 main components:
    * self.bert: the pretrained masked language model
    * self.backbone: wraps the bert model is a specific head
    * self.evaluator: handles computing the loss using the backbone and evaluate the metrics

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

    def __init__(
        self,
        *,
        tokenizer: Union[PreTrainedTokenizerFast, DictConfig],
        bert: Union[BertPreTrainedModel, DictConfig],
        module: Union[DictConfig, Module],
        head: Union[DictConfig, Module],
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        # `lr` and `weight_decay` are registered in .hparams
        self.save_hyperparameters()
        assert self.hparams["lr"] == lr
        assert self.hparams["weight_decay"] == weight_decay

        # instantiate the model
        self.module: Optional[Module] = maybe_instantiate(
            module,
            bert=bert,
            head=head,
            tokenizer=tokenizer,
            _recursive_=False,
        )

    def forward(self, batch: Batch, **kwargs) -> Batch:
        return self.module.forward(batch, **kwargs)

    def predict(self, batch: Batch, **kwargs) -> Batch:
        return self.module.predict(batch, **kwargs)

    def _step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: Optional[int],
        *,
        split: Split,
        **kwargs,
    ) -> Batch:
        """
        Perform the model forward pass and compute the loss or pre loss terms.
        !! This step is performed separately on each device. !!
        """
        return self.module.step(batch, **kwargs)

    def _step_end(self, pre_output: Batch, *, split: Split, log_data=True) -> Batch:
        """
        Call the `evaluator.forward_end` method (finalize the loss computation
        and update the metrics) using the `pre_output` data gathered from
        all devices.

        !! This step is performed on device 0 !!
        """
        output = self.module.step_end(pre_output, split)

        if log_data:
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
        Each key is formatted as: `prefix/key` where prefix is usually
        the split id.
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
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
            #configure-optimizers
        """
        return AdamW(
            params=only_trainable(self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

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
