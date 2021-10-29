from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import dill
from datasets import Dataset
from rich.progress import track

from fz_openqa.utils.datastruct import Batch


@dataclass
class SearchResult:
    score: List[List[float]]
    index: List[List[int]]


class Index:
    """Keep an index of a Dataset and search using queries."""

    index_name: Optional[str] = None
    is_indexed: bool = False
    params: Dict[str, Any] = {}

    def __init__(self, verbose: bool = False, **kwargs):
        self.params = {
            k: v
            for k, v in {**locals(), **kwargs}.items()
            if k not in ["self", "kwargs", "__class__"]
        }
        self.verbose = verbose

    def __repr__(self):
        params = self.params
        params["is_indexed"] = self.is_indexed
        params["index_name"] = self.index_name
        params = [f"{k}={v}" for k, v in params.items()]
        return f"{self.__class__.__name__}({', '.join(params)})"

    def new(self, **kwargs) -> "Index":
        params = self.params.copy()
        params.update(**kwargs)
        return type(self)(**params)

    def dill_inspect(self) -> bool:
        """check if the module can be pickled."""
        return dill.pickles(self)

    def build(self, dataset: Dataset, **kwargs):
        """Index a dataset."""
        raise NotImplementedError

    def search_one(
        self, query: Dict[str, Any], *, k: int = 1, **kwargs
    ) -> Tuple[List[float], List[int]]:
        """Search the index using one `query`"""
        raise NotImplementedError

    def search(self, query: Batch, *, k: int = 1, **kwargs) -> SearchResult:
        """Batch search the index using the `query` and
        return the scores and the indexes of the results
        within the original dataset.

        The default method search for each example sequentially.
        """
        batch_size = len(next(iter(query.values())))
        scores, indexes = [], []
        _iter = range(batch_size)
        if self.verbose:
            _iter = track(
                _iter,
                description=f"Searching {self.__class__.__name__} for batch..",
            )
        for i in _iter:
            eg = self.get_example(query, i)
            scores_i, indexes_i = self.search_one(eg, k=k, **kwargs)
            scores += [scores_i]
            indexes += [indexes_i]

        return SearchResult(index=indexes, score=scores)

    def get_example(self, query: Batch, index: int) -> Dict[str, Any]:
        return {k: v[index] for k, v in query.items()}
