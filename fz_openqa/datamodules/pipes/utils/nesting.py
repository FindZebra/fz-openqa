from typing import Any
from typing import Iterable
from typing import List
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
import torch
from torch import Tensor


def flatten_nested(values: List[List], level=1, current_level=0) -> Iterable[Any]:
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
    for x in values:
        if isinstance(x, list) and current_level < level:
            for y in flatten_nested(x, level, current_level + 1):
                yield y
        else:
            yield x


def infer_missing_dims(n_elements: int, *, shape: List[int]) -> List[int]:
    """
    Infer the missing dimensions in a shape.

    Parameters
    ----------
    n_elements
        The total number of elements
    shape
        Partial shape (e.g. (-1, 8, 6))

    Returns
    -------
    List[int]
        The inferred shape (e.g. (10, 8, 6))
    """
    assert all(y != 0 for y in shape)
    if all(y > 0 for y in shape):
        return shape
    else:
        neg_idx = [i for i, y in enumerate(shape) if y < 0]
        assert len(neg_idx) == 1, "Only one dimension can be negative"
        neg_idx = neg_idx[0]
        known_dims = [y for y in shape if y > 0]

        p = np.prod(known_dims)
        assert n_elements % p == 0, "n_elements must be divisible by product of known dimensions"
        missing_dim = n_elements // p
        shape[neg_idx] = missing_dim

    return shape


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
