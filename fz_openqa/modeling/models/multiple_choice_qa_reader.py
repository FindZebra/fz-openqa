import warnings
from typing import Dict

import torch
from torch import Tensor, nn
from transformers import (
    PreTrainedTokenizerFast,
    AutoModel,
    BertPreTrainedModel,
)

from fz_openqa.modeling.evaluators.abstract import Evaluator
from fz_openqa.modeling.functional import padless_cat
from fz_openqa.modeling.layers.lambd import Lambda
from .base import BaseModel


def flatten(x: Tensor) -> Tensor:
    return x.view(-1, x.shape[-1])


class cls_head(nn.Module):
    def __init__(self, bert: BertPreTrainedModel, output_size: int):
        super().__init__()
        self.linear = nn.Linear(bert.config.hidden_size, output_size)

    def forward(self, last_hidden_state: Tensor):
        cls_ = last_hidden_state[:, 0]  # CLS token
        return torch.nn.functional.normalize(self.linear(cls_), p=2, dim=-1)


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
        bert_id: str,
        evaluator: Evaluator,
        cache_dir: str,
        hidden_size: int = 256,
        dropout: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.tokenizer = tokenizer
        self.vocabulary_size = len(tokenizer.get_vocab())
        self.pad_token_id = tokenizer.pad_token_id

        # evaluator: compute the loss and the metrics to be logged in
        self.evaluator = evaluator

        # pretrained model
        self.bert: BertPreTrainedModel = AutoModel.from_pretrained(
            bert_id, cache_dir=cache_dir
        )
        self.bert.resize_token_embeddings(
            len(tokenizer)
        )  # necessary because of the added special tokens

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

        # compute contextualized representations
        ids = padless_cat(
            batch["document.input_ids"],
            batch["question.input_ids"],
            self.pad_token_id,
        )
        attn = padless_cat(
            batch["document.attention_mask"],
            batch["question.attention_mask"],
            self.pad_token_id,
        )
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
