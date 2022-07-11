from __future__ import annotations

import math
from copy import deepcopy
from functools import partial
from random import randint
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar

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

# Array2d = TypeVar('Array2d', List[List[Any], np.ndarray, Tensor)

Array2d = TypeVar("Array2d", list, np.ndarray, Tensor)
FillValue = TypeVar("FillValue", float, int, np.ndarray)

FLOAT_INF: float = 1e-18


def is_negative(x):
    return x < 0


def always_true(*args, **kwargs):
    return True


def pad_to_length(lst: List[FillValue], *, length: int, fill_token: FillValue) -> List[FillValue]:
    if len(lst) < length:
        if isinstance(lst, list):
            lst.extend([fill_token] * (length - len(lst)))
        elif isinstance(lst, np.ndarray):
            lst = np.pad(lst, (0, length - len(lst)), mode="constant", constant_values=fill_token)
        elif isinstance(lst, Tensor):
            lst = F.pad(lst, (0, length - len(lst)), value=fill_token)
        else:
            raise TypeError(f"Unsupported type: {type(lst)}")
    return lst[:length]


def pad_second_dim(arr: Array2d, *, k: int, fill_token: Any) -> Array2d:
    """
    Pad second dimension to length k.
    """

    if isinstance(arr, list):
        pad_fn = partial(pad_to_length, fill_token=fill_token, length=k)
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


def unique_second_dim(arr: Array2d, length: int, fill_token: Any = -1) -> Array2d:
    """
    Remove duplicate rows in a 2d array.
    """
    pad_fn = partial(pad_to_length, fill_token=fill_token, length=length)
    if isinstance(arr, list):
        return [pad_fn(list(set(row))) for row in arr]
    elif isinstance(arr, np.ndarray):
        return np.stack([pad_fn(np.unique(a)) for a in arr])
    elif isinstance(arr, Tensor):
        return torch.stack([pad_fn(torch.unique(a)) for a in arr])
    else:
        raise TypeError(f"Unsupported type: {type(arr)}")


def masked_fill(
    arr: Array, *, new_value: float | Tuple[int, int], condition: Optional[Callable] = None
) -> Array2d:
    """
    Replace all occurrences of replace_token in arr with new_token.
    """
    assert isinstance(new_value, (float, int, tuple))
    if condition is None:
        condition = always_true

    if isinstance(arr, list):
        has_neg_index = any(condition(i) for i in flatten_json_struct(arr))
        if has_neg_index:

            def _replace(x):
                if condition(x):
                    if isinstance(new_value, tuple):
                        return randint(*new_value)
                    else:
                        return new_value
                else:
                    return x

            return apply_to_json_struct(arr, _replace)
        else:
            return arr

    elif isinstance(arr, np.ndarray):
        if isinstance(new_value, tuple):
            rdn = np.random.randint(low=new_value[0], high=new_value[1], size=arr.shape)
        else:
            rdn = np.full_like(arr, new_value)
        return np.where(condition(arr), rdn, arr)
    elif isinstance(arr, Tensor):
        if isinstance(new_value, tuple):
            rdn = torch.randint_like(arr, low=new_value[0], high=new_value[1])
        else:
            rdn = torch.full_like(arr, new_value)
        return torch.where(condition(arr), rdn, arr)
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
        index: Array2d,
        score: Optional[Array2d] = None,
        tokens: Optional[List[List[str]]] = None,
        dataset_size: Optional[int] = None,
        format: Optional[OutputFormat] = None,
        fill_missing_values: bool = False,
        k: int,
    ):
        if score is None:
            score = masked_fill(deepcopy(index), new_value=0)
        else:
            if not all(len(x) <= k for x in score):
                raise ValueError(
                    f"All results must have length <= k. Found: {[len(x) for x in score]}"
                )

        # pad to length
        index = pad_second_dim(index, k=k, fill_token=-1)
        score = pad_second_dim(score, k=k, fill_token=-float("inf"))
        if tokens is not None:
            tokens = pad_second_dim(tokens, k=k, fill_token=[])

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

        # replace zero_index
        if fill_missing_values:
            if self.dataset_size is None:
                raise ValueError(
                    "`dataset_size` must be provided when `fill_missing_values` is True."
                )
            self.index = masked_fill(
                self.index, new_value=(0, self.dataset_size - 1), condition=is_negative
            )

    def __add__(self, other):
        if not isinstance(other, SearchResult):
            raise TypeError(f"Unsupported type: {type(other)}")

        self.score = concat_arrays(self.score, other.score)
        self.index = concat_arrays(self.index, other.index)
        if self.tokens is not None:
            self.tokens = concat_arrays(self.tokens, other.tokens)

        return self

    @property
    def rank(self):
        def rank_list(x):
            return [i for i, s in sorted(enumerate(x), key=lambda x: x[1], reverse=True)]

        if isinstance(self.score, list):
            return [rank_list(x) for x in self.score]
        elif isinstance(self.score, np.ndarray):
            return np.argsort(self.score, axis=-1)[..., ::-1]
        elif isinstance(self.score, Tensor):
            return self.score.argsort(dim=-1, descending=True)
        else:
            raise TypeError(f"Unsupported type: {type(self.score)}")

    def __repr__(self):
        score_shape = FormatArray(OutputFormat.NUMPY)(self.score).shape
        index_shape = FormatArray(OutputFormat.NUMPY)(self.index).shape
        return f"{type(self).__name__}(score={score_shape}, index={index_shape})"

    def _fill_rdn(self, args) -> Tuple[int, float]:
        """replace negative index values"""
        index, score = args
        if index < 0:
            return (randint(0, self.dataset_size - 1), -1)
        else:
            return (index, score)

    def to(self, output_format: Optional[OutputFormat] = None) -> "SearchResult":
        formatter = FormatArray(output_format)
        self.index = formatter(self.index)
        self.score = formatter(self.score)
        return self

    def union(self, other: "SearchResult", new_size: Optional[int] = None) -> "SearchResult":

        if not isinstance(other, SearchResult):
            raise TypeError(f"Unsupported type: {type(other)}")
        if len(self) != len(other):
            raise ValueError(
                f"Search results must have the same length, but {len(self)} != {len(other)}"
            )

        to_torch = FormatArray(OutputFormat.TORCH)
        total_size = new_size or self.k + other.k

        # sum the token scores for each pid
        new_score = []
        new_index = []
        for i in range(len(self.score)):
            # todo: propagate -inf
            # get scores for index i and substract the minimum values
            scores_a_i = to_torch(self.score[i])
            # is_inf = torch.isinf(scores_a_i)
            min_score_a_i = scores_a_i[~scores_a_i.isinf()].min()
            scores_a_i = scores_a_i - min_score_a_i
            scores_b_i = to_torch(other.score[i])
            # is_inf = is_inf | torch.isinf(scores_b_i)
            scores_b_i = scores_b_i - min_score_a_i
            min_score_b_i = scores_b_i[~scores_b_i.isinf()].min()
            scores_b_i = scores_b_i - min_score_b_i
            scores_i = torch.cat([scores_a_i, scores_b_i])

            # get indices for index i
            indices_a_i = to_torch(self.index[i])
            indices_b_i = to_torch(other.index[i])
            indices_i = torch.cat([indices_a_i, indices_b_i])

            # ge the unique indices
            unique_indices, u_inv = torch.unique(indices_i, return_inverse=True)
            unique_scores = torch.zeros_like(
                unique_indices, dtype=scores_i.dtype, device=scores_i.device
            )
            unique_scores += min_score_a_i + min_score_b_i
            unique_scores.index_add_(0, u_inv, scores_i)

            # sort the indices and scores
            unique_scores, idx = torch.sort(unique_scores, descending=True)
            unique_indices = unique_indices[idx]

            # pad / truncate the results
            unique_indices = pad_to_length(unique_indices, length=total_size, fill_token=-1)
            unique_scores = pad_to_length(unique_scores, length=total_size, fill_token=-math.inf)

            # append
            new_score.append(unique_scores)
            new_index.append(unique_indices)

        # stack and return
        new_score = torch.stack(new_score, dim=0)
        new_index = torch.stack(new_index, dim=0)

        return SearchResult(index=new_index, score=new_score, tokens=None, k=total_size)

    def __or__(self, other: "SearchResult") -> "SearchResult":
        return self.union(other)

    def __len__(self):
        return len(self.index)
