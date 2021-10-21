from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from datasets import Split
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from transformers import AdamW
from transformers import BertPreTrainedModel
from transformers import PreTrainedTokenizerFast

from fz_openqa.modeling.backbone import Backbone
from fz_openqa.modeling.evaluators.base import Evaluator
from fz_openqa.utils import maybe_instantiate
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import is_loggable
from fz_openqa.utils.functional import only_trainable


class Module(LightningModule):
    """
    This class implements the basics of evaluation, logging and inference using
    pytorch lightning mechanics.

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

    # features required for inference
    _required_feature_names = []
    # metrics that will be display in the progress bar
    _prog_bar_metrics = []

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        bert: Union[BertPreTrainedModel, DictConfig],
        backbone: Union[Backbone, DictConfig],
        evaluator: Optional[Union[DictConfig, Evaluator]],
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        # `lr` and `weight_decay` are registered in .hparams
        self.save_hyperparameters(ignore=["evaluator", "tokenizer", "bert"])
        assert self.hparams["lr"] == lr
        assert self.hparams["weight_decay"] == weight_decay

        # instantiate the pretrained language model
        self.bert = self.instantiate_bert(bert=bert, tokenizer=tokenizer)

        # instantiate the backbone model
        self.backbone = maybe_instantiate(backbone, bert=bert)

        # evaluator: compute the loss and take care of computing and logging the metrics
        self.evaluator: Optional[Evaluator] = maybe_instantiate(
            evaluator, backbone=backbone
        )

    def instantiate_bert(
        self,
        *,
        bert: Union[BertPreTrainedModel, DictConfig],
        tokenizer: PreTrainedTokenizerFast,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> BertPreTrainedModel:
        """Instantiate a bert model, and extend its embeddings to match the tokenizer"""

        self.vocabulary_size = len(tokenizer.get_vocab())
        self.pad_token_id = tokenizer.pad_token_id

        bert: BertPreTrainedModel = maybe_instantiate(
            bert, cache_dir=cache_dir, **kwargs
        )
        # extend BERT embeddings for the added special tokens
        # TODO: CRITICAL: check this does not affect the model
        #  this might explain the drop of performances
        bert.resize_token_embeddings(len(tokenizer))
        return bert

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
        return self.evaluator.step(batch, split=split)

    def _step_end(
        self, pre_output: Batch, *, split: Split, log_data=True
    ) -> Batch:
        """
        Call the `evaluator.forward_end` method (finalize the loss computation
        and update the metrics) using the `pre_output` data gathered from
        all devices.

        !! This step is performed on device 0 !!
        """
        output = self.evaluator.step_end(pre_output, split)

        if log_data:
            # potentially log the loss and
            # other metrics that are computed on each step
            self.log_data(output, prefix=f"{split}/")

        return output

    def _epoch_end(
        self, outputs: List[Any], *, split: Split, log_data=True
    ) -> Batch:
        """
        1. Compute the metrics for the whole epoch using `evaluator.compute_metrics`
        2. Log the metrics for the whole epoch
        """
        assert self.evaluator is not None
        metrics = self.evaluator.compute_metrics(split=split)
        if log_data:
            self.log_data(metrics, prefix=f"{split}/")
        self.evaluator.reset_metrics(split=split)
        return metrics

    def log_data(
        self,
        data: Batch,
        prefix: str = "",
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
            key = f"{prefix}{k}"
            if is_loggable(v):
                self.log(
                    key,
                    v,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=key in self._prog_bar_metrics,
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
    ) -> Dict[str, Any]:
        return self._step(batch, batch_idx, dataloader_idx, split=Split.TRAIN)

    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> Batch:
        return self._step(
            batch, batch_idx, dataloader_idx, split=Split.VALIDATION
        )

    def test_step(
        self,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ):
        return self._step(
            batch, batch_idx, dataloader_idx, split=Split.VALIDATION
        )

    def training_epoch_end(self, outputs: List[Any]):
        self._epoch_end(outputs, split=Split.TRAIN)

    def validation_epoch_end(self, outputs: List[Any]):
        self._epoch_end(outputs, split=Split.VALIDATION)

    def test_epoch_end(self, outputs: List[Any]):
        self._epoch_end(outputs, split=Split.TEST)

    def check_input_features(self, batch):
        for f in self._required_feature_names:
            assert f in batch.keys(), f"The feature {f} is required."
