from functools import partial
from random import randint
from typing import List
from typing import Optional
from typing import T
from typing import Tuple

import rich

from fz_openqa.datamodules.pipes.utils.nesting import nested_list


def pad_to_length(lst: List[T], *, length: int, fill_token: T) -> List[T]:
    if len(lst) < length:
        lst.extend([fill_token] * (length - len(lst)))
    return lst


class SearchResult:
    """A small class to help handling the search results."""

    def __init__(
        self,
        *,
        score: List[List[float]],
        index: List[List[int]],
        tokens: Optional[List[List[str]]] = None,
        dataset_size: Optional[int] = None,
        k: int,
    ):

        self.score = score
        self.index = index
        self.tokens = tokens
        self.dataset_size = dataset_size
        self.k = k

        # check batch lengths
        assert len(self.score) == len(self.index)
        if self.tokens:
            assert len(self.score) == len(self.tokens)

        # pad to length
        pad_fn = partial(pad_to_length, fill_token=-1, length=self.k)
        rich.print(self.index)
        self.index = list(map(pad_fn, self.index))
        pad_fn = partial(pad_to_length, fill_token=-1.0, length=self.k)
        self.score = list(map(pad_fn, self.score))
        if self.tokens is not None:
            pad_fn = partial(pad_to_length, fill_token=[], length=self.k)
            self.tokens = list(map(pad_fn, self.tokens))

        # replace zero_index
        if self.dataset_size is not None:
            has_neg_index = any(i < 0 for row in self.index for i in row)
            if has_neg_index:
                n_rows = len(self.score)
                flat_index = [i for row in self.index for i in row]
                flat_scores = [s for row in self.score for s in row]
                out = map(self._fill_rdn, zip(flat_index, flat_scores))
                flat_index, flat_scores = map(list, zip(*out))

                self.index = nested_list(flat_index, shape=[n_rows, -1])
                self.flat_scores = nested_list(flat_scores, shape=[n_rows, -1])

    def _fill_rdn(self, args) -> Tuple[int, float]:
        """replace negative index values"""
        index, score = args
        if index < 0:
            return (randint(0, self.dataset_size - 1), -1)
        else:
            return (index, score)
