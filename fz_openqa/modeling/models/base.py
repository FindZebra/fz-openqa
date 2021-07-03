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

from fz_openqa.modeling.evaluators.abstract import Evaluator
from fz_openqa.utils import maybe_instantiate
from fz_openqa.utils.utils import only_trainable


class BaseModel(LightningModule):
    _required_infer_feature_names = []
    _prog_bar_metrics = []  # metrics that will be display in the progress bar

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
        self.evaluator = maybe_instantiate(self.hparams.pop("evaluator"))

    def instantiate_bert(
        self,
        *,
        bert: Union[BertPreTrainedModel, DictConfig],
        tokenizer: PreTrainedTokenizerFast,
    ):
        """Instantiate a bert model as an attribute, and extend its embeddings to match the tokenizer"""

        self.vocabulary_size = len(tokenizer.get_vocab())
        self.pad_token_id = tokenizer.pad_token_id

        self.bert: BertPreTrainedModel = maybe_instantiate(bert)
        # extend BERT embeddings for the added special tokens
        self.bert.resize_token_embeddings(
            len(tokenizer)
        )  # TODO: CRITICAL: check this does not affect the model

    def _step(
        self, batch: Any, batch_idx: int, split: Split, log_data=True
    ) -> Dict[str, Any]:
        data = self.evaluator(self.forward, batch, split=split)

        if log_data:
            self.log_data(data, prefix=f"{split}/")

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return data

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        return self._step(batch, batch_idx, Split.TRAIN)

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        return self._step(batch, batch_idx, Split.VALIDATION)

    def test_step(self, batch: Any, batch_idx: int):
        return self._step(batch, batch_idx, Split.VALIDATION)

    def _epoch_end(self, outputs: List[Any], split: Split, log_data=True):
        # `outputs` is a list of dicts returned from `training_step()`
        assert self.evaluator is not None
        metrics = self.evaluator.compute_metrics(split=split)
        if log_data:
            self.log_data(metrics, prefix=f"{split}/")
        self.evaluator.reset_metrics(split=split)

    def training_epoch_end(self, outputs: List[Any]):
        return self._epoch_end(outputs, Split.TRAIN)

    def validation_epoch_end(self, outputs: List[Any]):
        return self._epoch_end(outputs, Split.VALIDATION)

    def test_epoch_end(self, outputs: List[Any]):
        return self._epoch_end(outputs, Split.TEST)

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
            self.log(
                f"{prefix}{k}",
                v,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=f"{prefix}{k}" in self._prog_bar_metrics,
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
