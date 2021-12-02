from abc import ABCMeta
from abc import abstractmethod
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from datasets import Dataset
from rich.progress import track

from fz_openqa.datamodules.component import Component
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.utils.datastruct import Batch


class IndexMode(Enum):
    INDEX = "index"
    QUERY = "query"


class Index(Component):
    """Keep an index of a Dataset and search it using queries."""

    # todo: cleanup the way attributes are set and build() is called (EsIndex)

    __metaclass__ = ABCMeta
    index_name: Optional[str] = None
    is_indexed: bool = False

    def __init__(
        self,
        dataset: Dataset,
        *,
        required_keys: List[str],
        query_field: str = "question",
        index_field: str = "document",
        index_name: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        dataset
            The dataset to be index (corpus)
        required_keys
            The keys required for each field. e.g. ['text']]
        query_field
            The field to be used as query (in the query dataset). e.g. question
        index_field
            The field to be used as index (in the dataset). e.g. document
        index_name:
            The name of the index. Must be unique to allow for sage caching.
        verbose
            Whether to print the progress
        kwargs
            Other parameters to be passed to the `build` method
        """
        super(Index, self).__init__(**kwargs)
        self.query_field = query_field
        self.index_field = index_field
        self.required_keys = required_keys
        self.index_name = index_name
        self.verbose = verbose
        self.dataset_size = len(dataset)
        self.build(dataset=dataset, **kwargs)

    def input_keys(self, mode: Optional[IndexMode] = None) -> List[str]:
        """Return the list of keys required for each mode (indexing, querying)"""
        output = []
        if mode is None or mode == IndexMode.INDEX:
            output += [f"{self.index_field}.{key}" for key in self.required_keys]

        if mode is None or mode == IndexMode.QUERY:
            output += [f"{self.query_field}.{key}" for key in self.required_keys]

        return output

    def __repr__(self):
        params = {"is_indexed": self.is_indexed, "index_name": self.index_name}
        params = [f"{k}={v}" for k, v in params.items()]
        return f"{self.__class__.__name__}({', '.join(params)})"

    @abstractmethod
    def build(self, dataset: Dataset, **kwargs):
        """Index a dataset."""
        raise NotImplementedError

    def search_one(
        self, query: Dict[str, Any], *, k: int = 1, **kwargs
    ) -> Tuple[List[float], List[int], Optional[List[int]]]:
        """Search the index using one `query`"""
        raise NotImplementedError

    def search(self, query: Batch, *, k: int = 1, **kwargs) -> SearchResult:
        """Batch search the index using the `query` and
        return the scores and the indexes of the results
        within the original dataset.

        The default method search for each example sequentially.
        """
        batch_size = len(next(iter(query.values())))
        scores, indexes, tokens = [], [], []
        _iter = range(batch_size)
        if self.verbose:
            _iter = track(
                _iter,
                description=f"Searching {self.__name__} for batch..",
            )
        for i in _iter:
            eg = self.get_example(query, i)
            scores_i, indexes_i, tokens_i = self.search_one(eg, k=k, **kwargs)
            scores += [scores_i]
            indexes += [indexes_i]
            if tokens_i is not None:
                tokens += [tokens_i]
        tokens = None if len(tokens) == 0 else tokens
        return SearchResult(
            index=indexes, score=scores, tokens=tokens, dataset_size=self.dataset_size, k=k
        )

    def get_example(self, query: Batch, index: int) -> Dict[str, Any]:
        return {k: v[index] for k, v in query.items()}
