from __future__ import annotations

import math
import warnings
from typing import Iterable
from typing import Optional

import rich
import torch
from torch import Tensor


def batch_cartesian_product(
    *inputs: Tensor, max_size: Optional[int] = None
) -> Iterable[Tensor | None]:
    """
    Cartesian product of a batch of tensors.

    Parameters
    ----------
    x
        tensor of shape (batch_size, n_vecs, n_dims, ...)
    max_size
        Maximum size of the resulting tensor,
        if the output size is larger than this values, sample `max_size` permutations at random
    Returns
    -------
    Tensor
     tensor of shape (batch_size, n_vecs, n_dims^n_vecs, ...)
    """
    x0, *xs = inputs
    bs, n_vecs, n_dims, *_ = x0.shape
    if n_vecs == 1:
        for y in (x0, *xs):
            yield y
        return
    if max_size is not None and max_size < n_dims:
        max_size = None
    index = torch.arange(n_dims, device=x0.device)
    index = torch.cartesian_prod(*(index for _ in range(n_vecs))).permute(1, 0)
    index_size = index.shape[1]

    # expand the index across the first dimension
    index = index[None, :, :].expand(bs, *index.shape)

    # resample the index if the output size is larger than max_size
    # Only the `max_size` max. values for the first input are used
    if max_size is not None and max_size < index_size:
        warnings.warn(
            f"The output size is larger than {max_size}, "
            f"taking the {max_size} top values according "
            f"to the first input"
        )
        x0_fp16 = x0.half()
        ref_scores = x0_fp16.gather(dim=2, index=index)
        ref_scores = ref_scores.sum(dim=1, keepdims=True)
        topk_indices = ref_scores.argsort(dim=-1, descending=True)

        topk_indices = topk_indices[..., :max_size]
        topk_indices = topk_indices.expand(-1, index.shape[1], topk_indices.shape[2])
        index = index.gather(index=topk_indices, dim=-1)

    # reindex all the inputs
    for y in (x0, *xs):
        if y is not None:
            bs, n_vecs, n_dims, *dims = y.shape
            _index = index.view(*index.shape, *(1 for _ in dims))
            _index = _index.expand(*index.shape, *dims)
            y = y.gather(dim=2, index=_index)
        yield y


def kl_divergence(p_logits: Tensor, q_logits: Optional[Tensor] = None, dim: int = -1) -> Tensor:
    log_p = p_logits.log_softmax(dim=dim)
    N = log_p.size(dim)
    if q_logits is None:
        log_q = -torch.ones_like(log_p) * math.log(N)
    else:
        log_q = q_logits.log_softmax(dim=dim)
    return (log_p.exp() * (log_p - log_q)).sum(dim=dim)
