from __future__ import annotations

import math
from typing import Iterable
from typing import Optional

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
    if max_size is not None and max_size < n_dims:
        max_size = None
    index = torch.arange(n_dims, device=x0.device)
    index = torch.cartesian_prod(*(index for _ in range(n_vecs))).permute(1, 0)
    if max_size is not None and max_size < index.shape[1]:
        perm = torch.randperm(index.size(1), device=x0.device)
        perm = perm[:max_size]
        perm = perm[None, :].expand(n_vecs, -1)
        index = index.gather(index=perm, dim=1)
    index = index[None, :, :].expand(bs, *index.shape)
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
