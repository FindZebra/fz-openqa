from __future__ import annotations

from typing import Iterable
from typing import List
from typing import Optional

import rich
import torch
from torch import Tensor

from fz_openqa.utils.datastruct import Batch


def check_only_first_doc_positive(
    batch: Batch, *, match_key="document.match_score", check_first_dim: bool = True
) -> None:
    if check_first_dim and not torch.all(batch[match_key][..., 0] > 0):
        msg = "Not all documents with index 0 are positive."
        raise ValueError(msg)
    if batch[match_key].shape[-1] > 1:
        if not torch.all(batch[match_key][..., 1:] == 0):
            raise ValueError("Not all documents with index >0 are negative.")


def expand_and_flatten(batch: Batch, n_docs, *, keys: List[str]) -> Batch:
    output = {}
    for k in keys:
        v = batch[k]
        v = v[:, None].expand(v.shape[0], n_docs, *v.shape[1:]).contiguous()
        output[k] = v.view(-1, *v.shape[2:])
    return output


def flatten_first_dims(batch: Batch, n_dims, *, keys: List[str]) -> Batch:
    """Collapse the first `n_dims` into a single dimension."""
    return {k: batch[k].view(-1, *batch[k].shape[n_dims:]) for k in keys}


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
