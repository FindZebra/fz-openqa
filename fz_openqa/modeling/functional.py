from collections.abc import Iterable
from typing import Sequence, Any, Union, Dict, Optional

import torch
from torch import Tensor

BatchValue = Union[Sequence[Tensor], Tensor]
Batch = Dict[str, BatchValue]


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


def pad(batch: BatchValue, pad_token: Any, length: Optional[int] = None):
    """pad a sequence of tensors to the same size, and remove the unnecessary right padding"""
    length = length or max(
        len(x) - count_right_padding(x, pad_token) for x in batch
    )

    def pad_one(max_length, x):
        padding = torch.tensor(
            max(0, max_length - len(x)) * [pad_token],
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat([x[:max_length], padding], dim=0).unsqueeze(0)

    padded = [pad_one(length, x) for x in batch]
    return torch.cat(padded, dim=0)


def is_valid_seq_attr(x: BatchValue, ref: BatchValue):
    """check that x is a sequence tensor of the same length as ref"""
    if isinstance(x, Tensor) and x.dim() == 1:
        return False

    def check_single(xx, xref):
        return hasattr(xx, "__len__") and len(xx) == len(xref)

    return all(check_single(xx, xref) for xx, xref in zip(x, ref))


def padless_cat(
    a: Batch,
    b: Batch,
    pad_token: Any,
    master_key: str = "input_ids",
    aux_pad_tokens: Optional[Dict[str, Any]] = None,
) -> Batch:
    """Concatenate the input tensors across the dimension 1 such that there is no padding between a and b."""
    if aux_pad_tokens is None:
        aux_pad_tokens = {}
    assert a.keys() == b.keys()
    assert master_key in a.keys()
    right_padding = [
        count_right_padding(xa, pad_token) for xa in a[master_key]
    ]
    output = {}
    keys = [master_key] + [k for k in a.keys() if k != master_key]

    def del_pad(x, p):
        if p > 0:
            return x[:-p]
        return x

    for key in keys:
        if key == master_key or (
            is_valid_seq_attr(a[key], a[master_key])
            and is_valid_seq_attr(b[key], b[master_key])
        ):
            batch_k = [
                torch.cat([del_pad(xa, rpad), xb], dim=0)
                for xa, xb, rpad in zip(a[key], b[key], right_padding)
            ]
            if key == master_key:
                output[key] = pad(batch_k, pad_token)
            else:
                length = int(output[master_key].shape[1])
                output[key] = pad(
                    batch_k, aux_pad_tokens.get(key, 0), length=length
                )
        else:
            err_msg = (
                f"Attempting to concatenate two batches, however"
                f"they differ on the non-sequential attribute with key={key}"
                f"with \na[key]={a[key]} \nb[key]=b[key]"
            )
            assert all(xa == xb for xa, xb in zip(a[key], b[key])), err_msg
            output[key] = a[key]

    return output


def flatten(x: Tensor) -> Tensor:
    return x.view(-1, x.shape[-1])
