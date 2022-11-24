from __future__ import annotations

from typing import Dict
from typing import Optional

import einops
import loguru
import rich
import torch
from omegaconf import DictConfig
from torch import nn
from torch import Tensor
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerFast
from warp_pipes import pprint_batch

from fz_openqa.modeling.heads import Head
from fz_openqa.modeling.modules.base import Module
from fz_openqa.utils import maybe_instantiate


class CrossAttentionHead(Head):
    """Score question and document using cross-attention."""

    id: str = "cross-attention"

    def __init__(
        self,
        *args,
        bias: bool = False,
        tokenizer: DictConfig | PreTrainedTokenizerFast,
        backbone: DictConfig | PreTrainedModel,
        **kwargs,
    ):
        super(CrossAttentionHead, self).__init__(*args, **kwargs)

        # backbone
        tokenizer = maybe_instantiate(tokenizer)
        self.cls_token_id = tokenizer.cls_token_id
        self.backbone = Module.instantiate_backbone(backbone=backbone, tokenizer=tokenizer)

        # projection
        self.bias = bias
        self.projection = nn.Linear(self.backbone.config.hidden_size, 1, bias=self.bias)

    def forward(
        self,
        *,
        batch: Dict[str, Tensor] = None,
        **kwargs,
    ) -> (Tensor, Dict):
        diagnostics = {}

        # get the tenors
        q_ids = batch["question.input_ids"]
        q_mask = batch["question.attention_mask"]
        d_ids = batch["document.input_ids"]
        d_mask = batch["document.attention_mask"]

        # remove CLS token and reshape Q
        bs, n_docs, *_ = d_ids.shape
        if self.cls_token_id is not None:
            if (q_ids[..., 0] == self.cls_token_id).all():
                q_ids = q_ids[..., 1:]
                q_mask = q_mask[..., 1:]

        q_ids = q_ids[:, None].expand(bs, n_docs, *q_ids.shape[1:])
        q_mask = q_mask[:, None].expand(bs, n_docs, *q_mask.shape[1:])

        # concatenate [D; Q] : vectors
        input_ids = torch.cat([d_ids, q_ids], dim=-1)
        attention_mask = torch.cat([d_mask, q_mask], dim=-1)

        # process using the backbone
        seq_len = input_ids.shape[-1]
        output = self.backbone(
            input_ids.view(-1, seq_len), attention_mask=attention_mask.view(-1, seq_len)
        )
        h_dq = output.last_hidden_state

        # compute the logits
        h_dq_cls = h_dq[..., 0, :]
        score = self.projection(h_dq_cls).squeeze(-1)
        diagnostics["score"] = score.view(bs, n_docs)

        return diagnostics

    def _preprocess(
        self, last_hidden_state: Tensor, head: str, mask: Optional[Tensor] = None, **kwargs
    ) -> Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement preprocessing")
