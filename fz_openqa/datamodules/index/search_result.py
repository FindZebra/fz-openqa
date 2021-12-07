from __future__ import annotations

from functools import partial
from random import randint
from typing import Any
from typing import List
from typing import Optional
from typing import T
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from fz_openqa.utils.array import Array
from fz_openqa.utils.array import concat_arrays
from fz_openqa.utils.array import FormatArray
from fz_openqa.utils.datastruct import OutputFormat
from fz_openqa.utils.json_struct import apply_to_json_struct
from fz_openqa.utils.json_struct import flatten_json_struct


Array2d = Union[List[List[Any]], np.ndarray, Tensor]


def pad_to_length(lst: List[T], *, length: int, fill_token: T) -> List[T]:
    if len(lst) < length:
        lst.extend([fill_token] * (length - len(lst)))
    return lst


def pad_second_dim(arr: Array2d, *, k: int, fill_token: Any) -> Array2d:
    """
    Pad second dimension to length k.
    """
    if isinstance(arr, list):
        pad_fn = partial(pad_to_length, fill_token=-1, length=k)
        return list(map(pad_fn, arr))
    elif isinstance(arr, np.ndarray):
        if arr.shape[1] == k:
            return arr
        return np.pad(
            arr, ((0, 0), (0, k - arr.shape[1])), mode="constant", constant_values=fill_token
        )
    elif isinstance(arr, Tensor):
        if arr.shape[1] == k:
            return arr
        return F.pad(arr, (0, k - arr.shape[1]), value=fill_token)
    else:
        raise TypeError(f"Unsupported type: {type(arr)}")


def replace_neg_values(arr: Array, *, new_value_range: Tuple[int, int]) -> Array2d:
    """
    Replace all occurrences of replace_token in arr with new_token.
    """
    if isinstance(arr, list):
        has_neg_index = any(i < 0 for i in flatten_json_struct(arr))
        if has_neg_index:

            def _replace(x):
                if x < 0:
                    return randint(*new_value_range)
                else:
                    return x

            return apply_to_json_struct(arr, _replace)
        else:
            return arr

    elif isinstance(arr, np.ndarray):
        rdn = np.random.randint(low=new_value_range[0], high=new_value_range[1], size=arr.shape)
        return np.where(arr < 0, rdn, arr)
    elif isinstance(arr, Tensor):
        rdn = torch.randint_like(arr, low=new_value_range[0], high=new_value_range[1])
        return torch.where(arr < 0, rdn, arr)
    else:
        raise TypeError(f"Unsupported type: {type(arr)}")


class SearchResult:
    """
    A small class to help handling the search results.
    I not enough results are returned, fill with negative indexes.
    If `dataset_size` is provided, replace negative values with random indexes.
    """

    def __init__(
        self,
        *,
        score: Array2d,
        index: Array2d,
        tokens: Optional[List[List[str]]] = None,
        dataset_size: Optional[int] = None,
        format: Optional[OutputFormat] = None,
        k: int,
    ):

        self.score = score
        self.index = index
        self.tokens = tokens
        self.dataset_size = dataset_size
        self.k = k

        formatter = FormatArray(format)
        self.score = formatter(self.score)
        self.index = formatter(self.index)

        # check batch lengths
        assert len(self.score) == len(self.index)
        if self.tokens:
            assert len(self.score) == len(self.tokens)

        # pad to length
        self.index = pad_second_dim(self.index, k=self.k, fill_token=-1)
        self.score = pad_second_dim(self.score, k=self.k, fill_token=-1.0)
        if self.tokens is not None:
            self.tokens = pad_second_dim(self.tokens, k=self.k, fill_token=[])

        # replace zero_index
        if self.dataset_size is not None:
            self.index = replace_neg_values(self.index, new_value_range=(0, self.dataset_size - 1))

    def __add__(self, other):
        if not isinstance(other, SearchResult):
            raise TypeError(f"Unsupported type: {type(other)}")

        self.score = concat_arrays(self.score, other.score)
        self.index = concat_arrays(self.index, other.index)
        if self.tokens is not None:
            self.tokens = concat_arrays(self.tokens, other.tokens)

        return self

    def __repr__(self):
        return f"{type(self).__name__}(score={self.score.shape}, index={self.index.shape})"

    def _fill_rdn(self, args) -> Tuple[int, float]:
        """replace negative index values"""
        index, score = args
        if index < 0:
            return (randint(0, self.dataset_size - 1), -1)
        else:
            return (index, score)
