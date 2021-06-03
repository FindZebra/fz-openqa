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

        # attention model for query-attention
        self._qkv = nn.Linear(
            bert_hdim, 3 * bert_hdim
        )  # the weights of the attention model
        self._attn = nn.MultiheadAttention(
            bert_hdim, num_heads=self.bert.config.num_attention_heads
        )  # attention layer
        self._proj = nn.Linear(bert_hdim, 1)

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
        ids = torch.cat([batch["document.input_ids"], batch["question.input_ids"]], dim=1)
        attn = torch.cat([batch["document.attention_mask"], batch["question.attention_mask"]], dim=1)
        # Todo: handle truncating in a nicer way (i.e. remove question padding first)
        heq = self.bert(ids[:, :512], attn[:, :512]).last_hidden_state  # [bs, L_e+L_q, h]
        ha = self.bert(
            flatten(batch["answer_choices.input_ids"]),
            flatten(batch["answer_choices.attention_mask"]),
        ).last_hidden_state  # [bs * N_a, L_a, h]

        # answer-question representation
        # here I use a full attention layer, even so this is quite a waste since we only use the output at position 0
        heq = self.expand_and_flatten(heq, N_a)  # [bs * N_a, L_q, h]
        hqa = torch.cat([heq, ha], dim=1).permute(1, 0, 2)  # [bs * N_a, L_q + L_a, h]
        hqa, _ = self._attn(*self._qkv(hqa).chunk(3, dim=-1))
        hqa_glob = self._proj(hqa[0, :, :])  # [bs * N_a, h]
        return hqa_glob.view(bs, N_a)

    @staticmethod
    def expand_and_flatten(x: Tensor, n: int) -> Tensor:
        """Expand a tensor of shape [bs, *dims] as [bs, n, *dims] and flatten to [bs * n, *dims]"""
        bs, *dims = x.shape
        x = x[:, None].expand(bs, n, *dims)
        x = x.contiguous().view(bs * n, *dims)
        return x
