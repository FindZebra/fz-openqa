from typing import *

import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerFast, AutoModel, BertPreTrainedModel

from src.modeling.evaluators import Evaluator
from .base import BaseModel


def flatten(x: Tensor) -> Tensor:
    return x.view(-1, x.shape[-1])


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
        bert_hdim = self.bert.config.hidden_size

        # projection heads
        self.e_proj = nn.Linear(bert_hdim, hidden_size)

        # attention model for query-attention
        self.qa_qkv = nn.Linear(
            bert_hdim, 3 * bert_hdim
        )  # the weights of the attention model
        self.qa_attn = nn.MultiheadAttention(
            bert_hdim, num_heads=self.bert.config.num_attention_heads
        )  # attention layer
        self.qa_proj = nn.Linear(bert_hdim, hidden_size)

    def e_repr(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        h = self.bert(input_ids, attention_mask).last_hidden_state
        h_cls = self.e_proj(h[:, 0])
        return h_cls

    def forward(self, batch: Dict[str, Tensor], **kwargs) -> torch.FloatTensor:
        """Compute the answer model p(a_i | q, e)"""
        for f in self._required_infer_feature_names:
            assert f in batch.keys(), f"The feature {f} is required for inference."

        # infer shapes
        bs, N_a, _ = batch["answer_choices.input_ids"].shape

        # compute contextualized representations
        he = self.bert(
            batch["document.input_ids"], batch["document.attention_mask"]
        ).last_hidden_state # [bs, L_e, h]
        hq = self.bert(
            batch["question.input_ids"], batch["question.attention_mask"]
        ).last_hidden_state # [bs, L_q, h]
        ha = self.bert(
            flatten(batch["answer_choices.input_ids"]),
            flatten(batch["answer_choices.attention_mask"]),
        ).last_hidden_state # [bs * N_a, L_a, h]

        # evidence representation
        he_glob = self.e_proj(he[:, 0]) # [bs, h]

        # answer-question representation
        hq = self.expand_and_flatten(hq, N_a) # [bs * N_a, L_q, h]
        hqa = torch.cat([hq, ha], dim=1).permute(1, 0, 2) # [bs * N_a, L_q + L_a, h]
        hqa, _ = self.qa_attn(*self.qa_qkv(hqa).chunk(3, dim=-1))
        hqa_glob = self.qa_proj(hqa[0, :, :]) # [bs * N_a, h]

        # compute the interaction model (dot-product) and return as shape [bs, N_a]
        # todo: local interaction model (using MaxSim or attention layer)
        he_glob = self.expand_and_flatten(he_glob, N_a)
        logits = (hqa_glob * he_glob).sum(1)
        return logits.view(bs, N_a)

    @staticmethod
    def expand_and_flatten(x: Tensor, n: int) -> Tensor:
        """Expand a tensor of shape [bs, *dims] as [bs, n, *dims] and flatten to [bs * n, *dims]"""
        bs, *dims = x.shape
        x = x[:, None].expand(bs, n, *dims)
        x = x.contiguous().view(bs * n, *dims)
        return x
