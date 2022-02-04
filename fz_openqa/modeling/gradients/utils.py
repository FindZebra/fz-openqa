from __future__ import annotations

from typing import Iterable
from typing import List
from typing import Optional

import torch
from torch import Tensor


def batch_cartesian_product(
    x: List[Tensor | None], max_size: Optional[int] = None
) -> Iterable[Tensor | None]:
    """
    Cartesian product of a batch of tensors.

    Parameters
    ----------
    x
        tensor of shape (batch_size, n_vecs, n_dims, ...)
    Returns
    -------
    Tensor
     tensor of shape (batch_size, n_vecs, n_dims^n_vecs, ...)
    """

    x0 = next(iter(y for y in x if y is not None))
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
    for y in x:
        if y is not None:
            bs, n_vecs, n_dims, *dims = y.shape
            _index = index.view(*index.shape, *(1 for _ in dims))
            _index = _index.expand(*index.shape, *dims)
            y = y.gather(dim=2, index=_index)
        yield y
