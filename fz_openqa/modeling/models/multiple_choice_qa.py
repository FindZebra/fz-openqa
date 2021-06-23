import re
from typing import Union, Any, Dict, List

from datasets import Split
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from transformers import AdamW
from transformers import (
    PreTrainedTokenizerFast,
    BertPreTrainedModel,
)

from .base import BaseModel


class MultipleChoiceQA(BaseModel):
    """A Multiple Choice model """

    _required_infer_feature_names = [
        "question.input_ids",
        "question.attention_mask",
        "question.input_ids",
        "document.attention_mask",
        "question.input_ids",
        "question.attention_mask",
        "answer_choices.input_ids",
        "answer_choices.attention_mask",
    ]
    _prog_bar_metrics = [
        "train/loss",
        "validation/loss",
        "validation/reader/Accuracy",
        "validation/retriever/Accuracy",
    ]  # metrics that will be display in the progress bar

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        bert: Union[BertPreTrainedModel, DictConfig],
        reader: Union[DictConfig, BaseModel],
        retriever: Union[DictConfig, BaseModel],
        **kwargs,
    ):
        super().__init__(**kwargs, evaluator=None)

        # instantiate the pretrained model
        self.instantiate_bert(bert=bert, tokenizer=tokenizer)

        # instantiate the reader
        self.reader = instantiate(reader, bert=self.bert, tokenizer=tokenizer)

        # instantiate the retriever
        self.retriever = instantiate(
            retriever, bert=self.bert, tokenizer=tokenizer
        )

    def _step(
        self,
        batch: Any,
        batch_idx: int,
        split: Split,
        log_data=True,
    ) -> Dict[str, Any]:

        reader_data = self.reader._step(
            batch, batch_idx, split, log_data=False
        )
        if log_data:
            self.log_data(reader_data, prefix=f"{split}/reader/")
        retriever_data = self.retriever._step(
            batch, batch_idx, split, log_data=False
        )
        if log_data:
            self.log_data(retriever_data, prefix=f"{split}/retriever/")

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": reader_data["loss"] + retriever_data["loss"]}

    def _epoch_end(self, outputs: List[Any], split: Split, log_data=True):
        # `outputs` is a list of dicts returned from `training_step()`
        reader_data = self.reader._epoch_end(outputs, split, log_data=False)
        if log_data:
            self.log_data(reader_data, prefix=f"{split}/reader/")
        retriever_data = self.retriever._epoch_end(
            outputs, split, log_data=False
        )
        if log_data:
            self.log_data(retriever_data, prefix=f"{split}/retriever/")

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        def filtered_params(model: nn.Module, pattern="^bert."):
            return (
                p
                for k, p in model.named_parameters()
                if not re.findall(pattern, k)
            )

        return AdamW(
            [
                {
                    "params": self.bert.parameters(),
                    "lr": float(self.hparams.lr),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": filtered_params(self.retriever),
                    "lr": float(self.retriever.hparams.lr),
                    "weight_decay": self.retriever.hparams.weight_decay,
                },
                {
                    "params": filtered_params(self.reader),
                    "lr": float(self.reader.hparams.lr),
                    "weight_decay": self.reader.hparams.weight_decay,
                },
            ],
        )
