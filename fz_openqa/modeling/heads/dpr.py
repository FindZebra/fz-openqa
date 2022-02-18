import math
from copy import deepcopy
from typing import Dict
from typing import Optional

import einops
import torch
import torch.nn.functional as F
from torch import einsum
from torch import nn
from torch import Tensor
from torch import unique

from fz_openqa.modeling.heads.base import Head
from fz_openqa.modeling.layers import BayesianLinear


class DprHead(Head):
    """Score question and document representations."""

    def __init__(
        self,
        *,
        across_batch: bool = False,
        normalize: bool = False,
        bias: bool = True,
        share_parameters: bool = False,
        bayesian: bool = False,
        **kwargs
    ):
        super(DprHead, self).__init__(**kwargs)
        self.across_batch = across_batch
        self.bias = bias

        Layer = nn.Linear if not bayesian else BayesianLinear

        self.normalize = normalize
        if self.output_size is not None:
            self.q_head = Layer(self.input_size, self.output_size, bias=self.bias)
            if share_parameters:
                self.d_head = self.q_head
            else:
                self.d_head = Layer(self.input_size, self.output_size, bias=self.bias)
        else:
            self.q_head = self.d_head = None

    def forward(
        self,
        *,
        hd: Tensor,
        hq: Tensor,
        doc_ids: Optional[Tensor] = None,
        q_mask: Optional[Tensor] = None,
        d_mask: Optional[Tensor] = None,
        batch: Dict[str, Tensor] = None,
        **kwargs
    ) -> Tensor:

        # preprocess
        hd = self.preprocess(hd, "document", mask=d_mask, batch=batch, **kwargs)
        hq = self.preprocess(hq, "question", mask=q_mask, batch=batch, **kwargs)

        # compute the score
        return self.score(hq=hq, hd=hd, doc_ids=doc_ids, batch=batch, **kwargs)

    def score(
        self,
        *,
        hq: Tensor,
        hd: Tensor,
        doc_ids: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        if not self.across_batch:
            return einsum("boh, bodh -> bod", hq, hd)
        else:
            hd = self._flatten_documents(hd, doc_ids)
            return einsum("boh, mh -> bom", hq, hd)

    def preprocess(
        self, last_hidden_state: Tensor, head: str, mask: Optional[Tensor] = None, **kwargs
    ) -> Tensor:
        cls_repr = last_hidden_state[..., 0, :]  # CLS token

        if self.output_size is not None:
            head = {"document": self.d_head, "question": self.q_head}[head]
            cls_repr = head(cls_repr)

        if self.normalize:
            cls_repr = F.normalize(cls_repr, p=2, dim=-1)
            # cls_repr /= float(cls_repr.shape[-1])**0.5

        return cls_repr

    @staticmethod
    def _flatten_documents(hd: Tensor, doc_ids=None) -> Tensor:
        if doc_ids is None:
            raise ValueError("doc_ids is required to compute the score across the batch")
        hd = einops.rearrange(hd, "bs opts docs ... -> (bs opts docs) ...")
        doc_ids = einops.rearrange(doc_ids, "bs opts docs -> (bs opts docs)")
        udoc_ids, uids = unique(doc_ids, return_inverse=True)
        hd = hd[uids]
        return hd
