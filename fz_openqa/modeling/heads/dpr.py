from __future__ import annotations

from copy import deepcopy
from typing import Dict
from typing import Optional

import einops
import rich
import torch
import torch.nn.functional as F
from torch import einsum
from torch import nn
from torch import Tensor
from torch import unique

from fz_openqa.modeling.heads.base import Head
from fz_openqa.modeling.layers import BayesianLinear


def unique_with_indices(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    uids, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return uids, inverse.new_empty(uids.size(0)).scatter_(0, inverse, perm)

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
        learn_scale: bool = False,
        scale: float = 1.0,
        auto_scale: bool = False,
        is_scaled: bool = False,
        scale_init: float = 1.0,
        **kwargs,
    ):
        super(DprHead, self).__init__(**kwargs)
        self.across_batch = across_batch
        self.bias = bias
        self.scale_init = scale
        self.register_buffer("is_scaled", torch.tensor(int(is_scaled)))
        self.auto_scale = auto_scale

        Layer = nn.Linear if not bayesian else BayesianLinear

        self.normalize = normalize

        self.share_parameters = share_parameters
        if self.output_size is not None:
            self.q_head = Layer(self.input_size, self.output_size, bias=self.bias)
            if share_parameters:
                self.d_head = self.q_head
            else:
                # NB: using deepcopy instead of initializing a new head is critical
                # to ensure stable training. Do not re-init the heads!
                self.d_head = deepcopy(self.q_head)
        else:
            self.q_head = self.d_head = None

        scale_value = torch.tensor(scale_init, dtype=torch.float)
        offset_value = torch.tensor(0.0, dtype=torch.float)
        if learn_scale:
            self._scale = nn.Parameter(scale_value)
            self._offset = nn.Parameter(offset_value)
        else:
            self.register_buffer("_scale", scale_value)
            self.register_buffer("_offset", offset_value)

    @property
    def scale_value(self):
        return self._scale

    def entropy(self) -> Tensor | float:
        if isinstance(self.q_head, BayesianLinear):
            entropy = self.q_head.entropy()
            if not self.share_parameters:
                entropy = entropy + self.d_head.entropy()
            return entropy

        return 0.0

    def kl(self) -> Tensor | float:
        if isinstance(self.q_head, BayesianLinear):
            kl = self.q_head.kl()
            if not self.share_parameters:
                kl = kl + self.d_head.kl()
            return kl

        return 0.0

    @property
    def offset(self):
        return self._offset

    def temperature(self) -> None:
        return self.scale_value.pow(-1)

    def set_scale(self, scores: Tensor):
        self._scale.data = self.scale_init * scores.std().detach().pow(-1) * self._scale.data
        self._offset.data = -scores.mean().detach() * self._scale.data
        self.is_scaled.data += 1

        scores = self.scale(scores)
        rich.print(
            f"> standardized | out.mean={scores.mean():.3f}, "
            f"out.std={scores.std():.3f}, "
            f"scale={self._scale.data:.3f}, "
            f"offset={self._offset.data:.3f}, "
            f"scaled={self.is_scaled}"
        )

        return scores

    def scale(self, hq: Tensor) -> Tensor:
        hq = hq * self.scale_value
        return hq

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

        # sample weights todo: only `if self.training`
        if isinstance(self.q_head, BayesianLinear):
            q_weights = self.q_head.sample_weights(hq)
            if not self.share_parameters:
                d_weights = self.d_head.sample_weights(hd)
            else:
                d_weights = q_weights
        else:
            q_weights = None
            d_weights = None

        # preprocess
        hd = self.preprocess(hd, "document", mask=d_mask, batch=batch, weights=d_weights, **kwargs)
        hq = self.preprocess(hq, "question", mask=q_mask, batch=batch, weights=q_weights, **kwargs)

        # compute the score
        score, diagnostics = self.score(hq=hq, hd=hd, doc_ids=doc_ids, batch=batch, **kwargs)

        if self.auto_scale and self.is_scaled < 1:
            score = self.set_scale(score)

        return score + self.offset, diagnostics

    def score(
        self,
        *,
        hq: Tensor,
        hd: Tensor,
        doc_ids: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        **kwargs,
    ) -> (Tensor, Dict):
        diagnostics = {}
        if not self.across_batch:
            return einsum("boh, bodh -> bod", hq, hd), diagnostics
        else:
            hd = self._flatten_documents(hd, doc_ids)
            return einsum("boh, mh -> bom", hq, hd), diagnostics

    def _preprocess(
        self,
        last_hidden_state: Tensor,
        head: str,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        head_kwargs = {"weights": weights} if weights is not None else {}
        cls_repr = last_hidden_state[..., 0, :]  # CLS token

        if self.output_size is not None:
            head = {"document": self.d_head, "question": self.q_head}[head]
            cls_repr = head(cls_repr, **head_kwargs)

        if self.normalize:
            cls_repr = F.normalize(cls_repr, p=2, dim=-1)

        if head == "question":
            cls_repr = self.scale(cls_repr)

        return cls_repr

    @staticmethod
    def _flatten_documents(hd: Tensor, doc_ids=None) -> Tensor:
        if doc_ids is None:
            raise ValueError("doc_ids is required to compute the score across the batch")
        hd = einops.rearrange(hd, "bs opts docs ... -> (bs opts docs) ...")
        doc_ids = einops.rearrange(doc_ids, "bs opts docs -> (bs opts docs)")
        udoc_ids, uids = unique_with_indices(doc_ids)
        hd = hd[uids]
        return hd
