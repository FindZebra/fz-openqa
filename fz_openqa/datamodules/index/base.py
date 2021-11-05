from abc import ABC
from abc import abstractmethod
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
    tokens: Optional[List[List[str]]] = None


class Index(ABC):
    """Keep an index of a Dataset and search using queries."""

    index_name: Optional[str] = None
    is_indexed: bool = False

    def __init__(self, dataset: Dataset, *, verbose: bool = False, **kwargs):
        self.verbose = verbose
        self.build(dataset=dataset, **kwargs)

    def __repr__(self):
        params = {"is_indexed": self.is_indexed, "index_name": self.index_name}
        params = [f"{k}={v}" for k, v in params.items()]
        return f"{self.__class__.__name__}({', '.join(params)})"

    def dill_inspect(self) -> bool:
        """check if the module can be pickled."""
        return dill.pickles(self)

    @abstractmethod
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
        scores, indexes, contents = [], [], []
        _iter = range(batch_size)
        if self.verbose:
            _iter = track(
                _iter,
                description=f"Searching {self.__class__.__name__} for batch..",
            )
        for i in _iter:
            eg = self.get_example(query, i)
            scores_i, indexes_i, contents_i = self.search_one(eg, k=k, **kwargs)
            scores += [scores_i]
            indexes += [indexes_i]
            contents += [contents_i]

        return SearchResult(index=indexes, score=scores, content=contents)

    def get_example(self, query: Batch, index: int) -> Dict[str, Any]:
        return {k: v[index] for k, v in query.items()}
