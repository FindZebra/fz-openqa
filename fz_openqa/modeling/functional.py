from collections.abc import Iterable
from typing import Sequence, Any, Union

import torch
from torch import Tensor

Batch = Union[Sequence[Tensor], Tensor]


def count_right_padding(
    x: Union[Sequence[Any], Tensor], pad_token: Any
) -> int:
    """count the number of right padding tokens"""
    if isinstance(x, Tensor):
        flipped_x = x.flip(0)
    elif isinstance(x, Iterable):
        flipped_x = x[::-1]
    else:
        raise NotImplementedError

    for i, t in enumerate(flipped_x):
        if t != pad_token:
            return i


def pad(batch: Batch, pad_token: Any):
    """pad a sequence of tensors to the same size, and remove the unnecessary right padding"""
    max_length = max(len(x) - count_right_padding(x, pad_token) for x in batch)

    def pad_one(max_length, x):
        padding = torch.tensor(
            max(0, max_length - len(x)) * [pad_token],
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat([x[:max_length], padding], dim=0).unsqueeze(0)

    padded = [pad_one(max_length, x) for x in batch]
    return torch.cat(padded, dim=0)


def padless_cat(a: Batch, b: Batch, pad_token: int) -> Tensor:
    """Concatenate the input tensors across the dimension 1 such that there is no padding between a and b."""
    batch = [
        torch.cat([xa[: -count_right_padding(xa, pad_token)], xb], dim=0)
        for xa, xb in zip(a, b)
    ]
    return pad(batch, pad_token)


def flatten(x: Tensor) -> Tensor:
    return x.view(-1, x.shape[-1])
