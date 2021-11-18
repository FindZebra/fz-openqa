from functools import partial
from typing import Any
from typing import Iterable
from typing import List
from typing import T
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
import torch
from torch import Tensor

from fz_openqa.utils.json_struct import apply_to_json_struct
from fz_openqa.utils.shape import infer_missing_dims
from fz_openqa.utils.shape import infer_shape


def flatten_nested_(values: List[List], level=1, current_level=0) -> Iterable[Any]:
    """
    Flatten a nested list of lists. See `flatten_nested` for more details.
    """
    for x in values:
        if isinstance(x, (list, np.ndarray, Tensor)) and current_level < level:
            for y in flatten_nested_(x, level, current_level + 1):
                yield y
        else:
            yield x


def flatten_nested(values: List[List], level=1) -> List[Any]:
    """
    Flatten a nested list of lists.

    Parameters
    ----------
    values
        The nested list to flatten.
    level
        The level of nesting to flatten.
    current_level
        The current level of nesting (don't set this manually).

    Returns
    -------
    Iterable[Any]
        flatten list of values.

    """
    return list(flatten_nested_(values, level=level))


def nested_list(values: List[Any], *, shape: Union[Tuple[int], List[int]], level=0) -> List[Any]:
    """
    Nest a list of values according to `shape`. This functions is similar to `np.reshape`.

    Parameters
    ----------
    values
        The values to be nested.
    shape
        The target shape
    level
        Do not use. Used for debugging.

    Returns
    -------
    List[Any]
        The nested list.
    """
    if not isinstance(shape, list):
        shape = list(shape)
    shape = infer_missing_dims(len(values), shape=shape)
    stride = int(np.prod(shape[1:]))
    output = []
    for j in range(0, len(values), stride):
        section = values[j : j + stride]
        if len(shape) > 2:
            output += [nested_list(section, shape=shape[1:], level=level + 1)]
        else:
            output += [section]

    return output


def reconcat(values: List[Any], original_type: Type):
    if original_type == Tensor:
        values = torch.cat([t[None] for t in values], dim=0)
    elif original_type == np.ndarray:
        values = np.concatenate([t[None] for t in values], dim=0)
    elif original_type == list:
        pass
    else:
        raise ValueError(f"Cannot reconstruct values of original type={original_type}")
    return values


def expand_and_repeat(x, n):
    """Expand by one dimension and repeat `n` times"""
    if isinstance(x, np.ndarray):
        x = x[..., None]
        return np.repeat(x, n, axis=-1)

    elif isinstance(x, torch.Tensor):
        x = x[..., None]
        return x.expand(x.shape[:-1] + (n,))

    elif isinstance(x, list):

        def repeat(x, *, n):
            return [x] * n

        return apply_to_json_struct(x, partial(repeat, n=n))


def expand_to_shape(x: T, target_shape: List[int]) -> T:
    """Expand a batch value to a target shape."""
    shape = infer_shape(x)

    # replace negative target_shape values with the original shape
    for i, s in enumerate(shape):
        if target_shape[i] < 0:
            target_shape[i] = s

    while len(shape) < len(target_shape):
        if not list(target_shape[: len(shape)]) == list(shape):
            raise ValueError(
                f"First dimentions must match. Cannot expand batch of "
                f"shape {shape} to shape {target_shape}"
            )

        # expand the batch by one dim
        new_dim = target_shape[len(shape)]
        x = expand_and_repeat(x, new_dim)

        # update `shape`
        shape = infer_shape(x)

    return x
