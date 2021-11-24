from random import randint
from typing import List
from typing import Optional
from typing import Tuple

from fz_openqa.datamodules.pipes.utils.nesting import nested_list


class SearchResult:
    """A small class to help handling the search results."""

    def __init__(
        self,
        *,
        score: List[List[float]],
        index: List[List[int]],
        tokens: Optional[List[List[str]]] = None,
        dataset_size: Optional[int] = None
    ):

        self.score = score
        self.index = index
        self.tokens = tokens
        self.dataset_size = dataset_size

        # check lengths
        assert len(self.score) == len(self.index)
        if self.tokens:
            assert len(self.score) == len(self.tokens)

        # replace zero_index
        if self.dataset_size is not None:
            has_neg_index = any(i < 0 for row in self.index for i in row)
            if has_neg_index:
                n_rows = len(self.score)
                flat_index = [i for row in self.index for i in row]
                flat_scores = [s for row in self.score for s in row]
                out = map(self._fill, zip(flat_index, flat_scores))
                flat_index, flat_scores = map(list, zip(*out))

                self.index = nested_list(flat_index, shape=[n_rows, -1])
                self.flat_scores = nested_list(flat_scores, shape=[n_rows, -1])

    def _fill(self, args) -> Tuple[int, float]:
        """replace negative index values"""
        index, score = args
        if index < 0:
            return (randint(0, self.dataset_size - 1), -1)
        else:
            return (index, score)
