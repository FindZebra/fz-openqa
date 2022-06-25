from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict
from typing import Optional

import rich
import torch
import torch.nn.functional as F
from torch import einsum
from torch import nn
from torch import Tensor

from fz_openqa.modeling.heads.base import Head
from fz_openqa.modeling.layers import BayesianLinear
from fz_openqa.utils.metric_type import MetricType


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
    uids, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return uids, inverse.new_empty(uids.size(0)).scatter_(0, inverse, perm), inverse


class DprHead(Head):
    """Score question and document representations."""

    def __init__(
        self,
        *,
        normalize: bool = False,
        bias: bool = False,
        share_parameters: bool = True,
        bayesian: bool = False,
        learn_scale: bool = False,
        target_scale_init: float = 1.0,
        auto_scale: bool = True,
        scale_init: float = 1.0,
        **kwargs,
    ):
        super(DprHead, self).__init__(**kwargs)
        self.bias = bias
        self.target_scale_init = target_scale_init
        self.register_buffer("is_scaled", torch.tensor(int(not auto_scale)))

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

        # set the temperature
        self._kappa = nn.Parameter(torch.tensor(0.0), requires_grad=learn_scale)
        self._kappa_zero = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.scale_value = scale_init

    @property
    def temperature(self) -> Tensor:
        return (self._kappa - self._kappa_zero).exp()

    @temperature.setter
    def temperature(self, value: Tensor):
        self._kappa.data.fill_(0)
        self._kappa_zero.data.fill_(-math.log(value))

    @property
    def scale_value(self) -> Tensor:
        return self.temperature.pow(-1)

    @scale_value.setter
    def scale_value(self, value: Tensor):
        self.temperature = 1 / value

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

    def set_scale(self, scores: Tensor, qmask: Optional[Tensor] = None):

        scores_std = self._scores_std(scores)

        self.scale_value = self.target_scale_init * scores_std.pow(-1)
        self.is_scaled.data += 1

        scores = self.scale_query(scores, qmask=qmask)
        rich.print(
            f"> standardized | out.mean={scores.mean():.3f}, "
            f"out.std={self._scores_std(scores):.3f}, "
            f"scale={self.scale_value:.3f}, "
            f"kappa={self._kappa.data:.3f}, "
            f"kappa_zero={self._kappa_zero.data:.3f}, "
            f"scaled={self.is_scaled}"
        )

        return scores

    def _scores_std(self, scores):
        scores_ = scores.view(-1).detach()
        scores_ = scores_[(~scores_.isnan()) & (~scores_.isinf())]
        scores_std = scores_.std()
        return scores_std

    def scale_query(self, hq: Tensor, qmask=None) -> Tensor:
        hq = hq / self.temperature
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
        diagnostics["score"] = score + self.offset

        return diagnostics

    def score(
        self,
        *,
        hq: Tensor,
        hd: Tensor,
        doc_ids: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict:
        diagnostics = {}
        vdim = hq.shape[-1]
        n_docs = hd.shape[-2]
        bs = hq.shape[:-1]
        shared_batch = self._is_shared_batch_dims(hd=hd, hq=hq, bs=bs, expected_hd_dim=2)

        # reshape (flatten)
        hq = hq.view(-1, vdim)

        if shared_batch:
            hd = hd.view(-1, n_docs, vdim)
            if self.metric_type == MetricType.inner_product:
                scores = einsum("bh, bdh -> bd", hq, hd)
            elif self.metric_type == MetricType.euclidean:
                _hq = hq[:, None, :]
                _hd = hd[:, :, :]
                scores = -1 * (_hq - _hd).pow(2).sum(-1).pow(0.5)
            else:
                raise ValueError(f"Unknown `metric_type`: {self.metric_type}")
        else:
            if self.metric_type == MetricType.inner_product:
                scores = einsum("bh, dh -> bd", hq, hd)
            elif self.metric_type == MetricType.euclidean:
                _hq = hq[:, None, :]
                _hd = hd[None, :, :]
                scores = -1 * (_hq - _hd).pow(2).sum(-1).pow(0.5)
            else:
                raise ValueError(f"Unknown `metric_type`: {self.metric_type}")

        # reshape and return
        scores = scores.view(*bs, n_docs)
        return scores, diagnostics

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
            cls_repr = self.scale_query(cls_repr)

        return cls_repr

    @staticmethod
    def _flatten_documents(hd: Tensor, *, doc_ids) -> (Tensor, Tensor):
        if doc_ids is None:
            raise ValueError("doc_ids is required to compute the score across the batch")
        feature_dimensions = len(doc_ids.shape)
        hd = hd.view(-1, *hd.shape[feature_dimensions:])
        doc_ids = doc_ids.view(-1)
        udoc_ids, uids, _ = unique_with_indices(doc_ids)
        hd = hd[uids]
        return hd, udoc_ids

    def _is_shared_batch_dims(self, *, hd, hq, bs, expected_hd_dim):
        if hq.shape[: len(bs)] == hd.shape[: len(bs)]:
            shared_batch = True
        else:
            if not len(hd.shape) == expected_hd_dim:
                raise ValueError(
                    f"hd: {hd.shape} and hq: {hq.shape} do not share the same batch size: {bs}. "
                    f"In that case, the dimension of hd should be {expected_hd_dim} "
                    f"(found {len(hd.shape)})."
                )
            shared_batch = False
        if hq.shape[-1] != hd.shape[-1]:
            raise ValueError(
                f"Question and documents don't share the same vector dimension:"
                f"Found: hq.shape={hq.shape}, hd.shape={hd.shape}"
            )
        return shared_batch
