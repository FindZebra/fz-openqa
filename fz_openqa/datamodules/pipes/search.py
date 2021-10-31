from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import dill
import torch
from datasets import Dataset
from pyarrow import Table

from . import Collate
from . import Nest
from .base import Pipe
from fz_openqa.utils.datastruct import Batch


@dataclass
class SearchResult:
    score: List[List[float]]
    index: List[List[int]]


class FakeIndex:
    """A small class to test Search corpus without using a proper index"""

    index_name = "<name>"

    def search(self, *, query: Batch, k: int, **kwargs) -> SearchResult:
        values = query["question.text"]
        return SearchResult(
            index=[[0 for _ in range(k)] for _ in values],
            score=[[1.0 for _ in range(k)] for _ in values],
        )

    def dill_inspect(self) -> bool:
        """check if the module can be pickled."""
        return dill.pickles(self)


class FakeDataset:
    """A small class to test Search corpus without using a proper index"""

    def __init__(self):
        self.data = {"document.text": "<text>", "document.row_idx": 0}

    def __getitem__(self, idx):
        """check if the module can be pickled."""
        if isinstance(idx, str):
            return [deepcopy(self.data)]
        else:
            return deepcopy(self.data)


class SearchCorpus(Pipe):
    """Search a Corpus object given a query"""

    def __init__(
        self,
        corpus_index,
        *,
        k: Optional[int] = None,
        model: Optional[Union[Callable, torch.nn.Module]] = None,
        index_output_key: str = "document.row_idx",
        score_output_key: str = "document.retrieval_score",
        **kwargs,
    ):
        super(SearchCorpus, self).__init__(**kwargs)
        self.index = corpus_index
        self.index_output_key = index_output_key
        self.score_output_key = score_output_key
        self.k = k
        self.model = model

    def __repr__(self):
        return {
            "__type__": type(self),
            "k": self.k,
            "es_index": self.index.index_name,
        }.__repr__()

    def dill_inspect(self, **kwargs) -> Dict[str, Any]:
        return {
            "__all__": dill.pickles(self),
            "index": self.index.dill_inspect(),
        }

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "__all__": self._fingerprint(self),
            "index": self._fingerprint(self.index),
        }

    def __call__(
        self,
        query: Batch,
        *,
        k: Optional[int] = None,
        model: Optional[Union[Callable, torch.nn.Module]] = None,
        simple_collate: Optional[bool] = None,
        **kwargs,
    ):
        # update args
        k = k or self.k
        model = model or self.model

        # query the index
        search_result = self.index.search(
            query=query, k=k, model=model, **kwargs
        )

        # retrieve the examples from the dataset (flat list)
        flat_indexes = (idx for sub in search_result.index for idx in sub)
        flat_scores = (score for sub in search_result.score for score in sub)
        examples = [
            {self.index_output_key: idx, self.score_output_key: score}
            for idx, score in zip(flat_indexes, flat_scores)
        ]

        # nest the examples:
        # [eg for eg in examples] -> [[eg_q for eg_q in results[q] for q in query]
        return Nest(stride=k)(Collate()(examples))


class FeatchDocuments(Pipe):
    def __init__(
        self,
        *,
        corpus_dataset: Dataset,
        keys: Optional[List[str]] = None,
        collate_pipe: Pipe = Collate(),
        index_key: str = "document.row_idx",
        id: str = "fetch-documents-pipe",
        **kwargs,
    ):
        super(FeatchDocuments, self).__init__(id=id)
        if keys is not None:
            corpus_dataset.remove_columns(
                set(corpus_dataset.column_names) - set(keys)
            )
        self.corpus_dataset = corpus_dataset
        self.keys = keys
        self.collate_pipe = collate_pipe
        self.index_key = index_key

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return self.corpus_dataset.column_names

    def __call__(self, batch: Batch, **kwargs) -> Union[List[Dict], Batch]:
        # todo: check dataset fingerprint
        # todo: get table in the __init__, removing dependencies
        #  on `datasets`might solve issues

        # get the `dataset` indexes
        indexes = [int(idx) for idx in batch[self.index_key]]

        err_msg = (
            "There is a mismatch between the query indexes and the retrieved indexes, "
            "make sure you are using the same dataset."
        )
        # fetch documents
        if len(indexes) < 500:
            rows: List[Dict] = [self.corpus_dataset[idx] for idx in indexes]
            assert indexes[0] == rows[0][self.index_key], err_msg
        # rows = self.corpus_dataset.select(indexes, keep_in_memory=True)
        else:
            rows: Table = self.corpus_dataset._data.table.take(indexes)
            rows: Batch = rows.to_pydict()
            assert indexes[0] == rows[self.index_key][0], err_msg

        # collate and return
        return self.collate_pipe(rows)
