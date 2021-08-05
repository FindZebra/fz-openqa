from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from datasets import Split
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor
from transformers import AdamW
from transformers import BertPreTrainedModel
from transformers import PreTrainedTokenizerFast

from fz_openqa.modeling.evaluators.base import Evaluator
from fz_openqa.utils import maybe_instantiate
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import is_loggable
from fz_openqa.utils.functional import only_trainable


class BaseModel(LightningModule):
    # features required for inference
    _required_infer_feature_names = []
    # metrics that will be display in the progress bar
    _prog_bar_metrics = []
    # prefix for the logged metrics
    # all metrics are of the form "split/_logging_prefix/metric_name"
    _logging_prefix = ""

    def __init__(
        self,
        *,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        evaluator: Optional[Evaluator] = None,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        # evaluator: compute the loss and the metrics to be logged in
        self.evaluator: Evaluator = maybe_instantiate(
            self.hparams.pop("evaluator")
        )

    def instantiate_bert(
        self,
        *,
        bert: Union[BertPreTrainedModel, DictConfig],
        tokenizer: PreTrainedTokenizerFast,
        cache_dir: Optional[str] = None,
    ):
        """Instantiate a bert model as an attribute, and extend its embeddings to match the tokenizer"""

        self.vocabulary_size = len(tokenizer.get_vocab())
        self.pad_token_id = tokenizer.pad_token_id

        self.bert: BertPreTrainedModel = maybe_instantiate(
            bert, cache_dir=cache_dir
        )
        # extend BERT embeddings for the added special tokens
        # TODO: CRITICAL: check this does not affect the model
        #  this might explain the drop of performances
        self.bert.resize_token_embeddings(len(tokenizer))

    def _step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int],
        *,
        split: Split,
        **kwargs,
    ) -> Batch:
        """This step is performed separately on each device.
        Do the forward pass and compute the loss.
        """
        if "input_ids" in batch.keys():
            # todo: temporary for the index_corpus callback
            return self.predict_step(batch, batch_idx, dataloader_idx)

        output = self.evaluator(self.forward, batch, split=split)
        return output

    def _step_end(self, output: Batch, *, split, log_data=True) -> Batch:
        """At this step, the output is gathered from all devices.
        Update the metrics and average the loss."""
        output = self.evaluator.forward_end(output, split)

        if log_data:
            # potentially log the loss and
            # other metrics that are computed on each step
            self.log_data(output, prefix=f"{split}/")

        return output

    def _epoch_end(
        self, outputs: List[Any], *, split: Split, log_data=True
    ) -> Batch:
        # `outputs` is a list of dicts returned from `training_step()`
        assert self.evaluator is not None
        metrics = self.evaluator.compute_metrics(split=split)
        if log_data:
            self.log_data(metrics, prefix=f"{split}/")
        self.evaluator.reset_metrics(split=split)
        return metrics

    def log_data(
        self,
        data: Dict[str, Tensor],
        prefix: str = "",
        on_step=False,
        on_epoch=True,
        sync_dist=True,
    ):
        # log 'split' metrics
        for k, v in data.items():
            key = f"{prefix}{self._logging_prefix}{k}"
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
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
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
