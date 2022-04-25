from __future__ import annotations

from typing import Dict
from typing import Optional

import loguru
import torch
from torch import nn
from torch import Tensor

from fz_openqa.modeling.heads import Head


class CrossAttentionHead(Head):
    """Score question and document using cross-attention."""

    id: str = "cross-attention"

    def __init__(self, *args, bias: bool = False, **kwargs):
        super(CrossAttentionHead, self).__init__(*args, **kwargs)
        self.bias = bias
        self.projection = nn.Linear(self.input_size, 1, bias=self.bias)

    def forward(
        self,
        *,
        hd: Tensor,
        hq: Tensor,
        doc_ids: Optional[Tensor] = None,
        q_mask: Optional[Tensor] = None,
        d_mask: Optional[Tensor] = None,
        batch: Dict[str, Tensor] = None,
        **kwargs,
    ) -> (Tensor, Dict):
        diagnostics = {}

        # concatenate [D; Q] : vectors
        bs, n_opts, n_docs, *_ = hd.shape
        hq = hq.unsqueeze(2).expand(bs, n_opts, n_docs, *hq.shape[2:])
        h_dq = torch.cat([hd, hq], dim=-2)

        # concatenate [D; Q] : attention_mask
        q_mask = batch["question.attention_mask"]
        d_mask = batch["document.attention_mask"]
        q_mask = q_mask.unsqueeze(2).expand(bs, n_opts, n_docs, *q_mask.shape[2:])
        dq_mask = torch.cat([d_mask, q_mask], dim=-1)

        # process with the bert layers
        if self.bert_layers is None:
            loguru.logger.error(
                f"No BERT layers were found in {type(self)}. " f"Cross Attention cannot be used."
            )
        h_dq = self._process_with_bert_layers(dq_mask, h_dq)

        # compute the logits
        h_dq_cls = h_dq[..., 0, :]
        score = self.projection(h_dq_cls).squeeze(-1)

        return score, diagnostics

    def _preprocess(
        self, last_hidden_state: Tensor, head: str, mask: Optional[Tensor] = None, **kwargs
    ) -> Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement preprocessing")
