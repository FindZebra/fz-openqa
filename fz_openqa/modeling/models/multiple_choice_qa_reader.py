import warnings
from typing import Dict, Optional, Union

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from transformers import (
    PreTrainedTokenizerFast,
    BertPreTrainedModel,
)

from fz_openqa.modeling.evaluators.abstract import Evaluator
from fz_openqa.modeling.functional import padless_cat, flatten
from fz_openqa.modeling.layers.heads import cls_head
from fz_openqa.modeling.models.base import BaseModel


class MultipleChoiceQAReader(BaseModel):
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
        "train/Accuracy",
        "validation/Accuracy",
    ]  # metrics that will be display in the progress bar

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        bert: Union[BertPreTrainedModel, DictConfig],
        evaluator: Union[Evaluator, DictConfig],
        cache_dir: Optional[str] = None,
        hidden_size: int = 256,
        dropout: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs, evaluator=evaluator)

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        # instantiate the pretrained model
        self.instantiate_bert(bert=bert, tokenizer=tokenizer)

        # heads
        self.qd_head = cls_head(self.bert, hidden_size)
        self.a_head = cls_head(self.bert, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: Dict[str, Tensor], **kwargs) -> torch.FloatTensor:
        """Compute the answer model p(a_i | q, e)"""
        for f in self._required_infer_feature_names:
            assert (
                f in batch.keys()
            ), f"The feature {f} is required for inference."

        # infer shapes
        bs, N_a, _ = batch["answer_choices.input_ids"].shape

        # concatenate questions and documents such that there is no padding between Q and D
        # todo: handle both input_ids and attention in one call
        #  not sure it works that way, potential alignment issue
        ids = padless_cat(
            batch["document.input_ids"],
            batch["question.input_ids"],
            self.pad_token_id,
        )
        attn = padless_cat(
            batch["document.attention_mask"],
            batch["question.attention_mask"],
            0,
        )

        # compute contextualized representations
        if len(ids) > 512:
            warnings.warn("the tensor [question; document] was truncated.")
        heq = self.bert(
            ids[:, :512], attn[:, :512]
        ).last_hidden_state  # [bs, L_e+L_q, h]
        ha = self.bert(
            flatten(batch["answer_choices.input_ids"]),
            flatten(batch["answer_choices.attention_mask"]),
        ).last_hidden_state  # [bs * N_a, L_a, h]

        # answer-question representation
        heq = self.qd_head(heq)  # [bs, h]
        ha = self.a_head(ha).view(bs, N_a, self.hparams.hidden_size)
        return torch.einsum("bh, bah -> ba", heq, ha)  # dot-product model

    @staticmethod
    def expand_and_flatten(x: Tensor, n: int) -> Tensor:
        """Expand a tensor of shape [bs, *dims] as [bs, n, *dims] and flatten to [bs * n, *dims]"""
        bs, *dims = x.shape
        x = x[:, None].expand(bs, n, *dims)
        x = x.contiguous().view(bs * n, *dims)
        return x
