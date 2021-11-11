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

from .base import Pipe
from .collate import Collate
from fz_openqa.utils.datastruct import Batch


@dataclass
class SearchResult:
    score: List[List[float]]
    index: List[List[int]]
    tokens: List[List[str]]


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
        analyzed_output_key: str = "document.analyzed_tokens",
        **kwargs,
    ):
        super(SearchCorpus, self).__init__(**kwargs)
        self.index = corpus_index
        self.index_output_key = index_output_key
        self.score_output_key = score_output_key
        self.analyzed_output_key = analyzed_output_key
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

    def _call(
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
        search_result = self.index.search(query, k=k, model=model, **kwargs)

        # store as a dictionary and return
        output = {
            self.index_output_key: search_result.index,
            self.score_output_key: search_result.score,
        }

        if search_result.tokens is not None:
            output[self.analyzed_output_key] = search_result.tokens

        return output


class FetchDocuments(Pipe):
    def __init__(
        self,
        *,
        corpus_dataset: Dataset,
        keys: Optional[List[str]] = None,
        collate_pipe: Pipe = None,
        index_key: str = "document.row_idx",
        output_format: str = "dict",
        id: str = "fetch-documents-pipe",
        **kwargs,
    ):
        super(FetchDocuments, self).__init__(id=id)
        if keys is not None:
            keys = set(keys).union([index_key])
            # make sure to sort the keys to ensure deterministic fingerprinting
            cols_to_drop = list(sorted(set(corpus_dataset.column_names) - keys))
            corpus_dataset = corpus_dataset.remove_columns(cols_to_drop)

        self.corpus_dataset = corpus_dataset
        self.keys = keys
        self.collate_pipe = collate_pipe or Collate()
        self.index_key = index_key
        self.output_format = output_format

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "__all__": self._fingerprint(self),
            "corpus_dataset": self._fingerprint(self.corpus_dataset),
            "corpus_dataset_fingerprint": self.corpus_dataset._fingerprint,
            "self.collate_pipe": self._fingerprint(self.collate_pipe),
        }

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return self.corpus_dataset.column_names

    def _call(self, batch: Batch, **kwargs) -> Batch:
        # todo: check dataset fingerprint (checking 1st index for now)

        # get the `dataset` indexes
        indexes = [int(idx) for idx in batch[self.index_key]]

        err_msg = (
            "There is a mismatch between the query indexes and the retrieved indexes, "
            "make sure you are using the same dataset."
        )

        # fetch documents
        table = self.corpus_dataset.select(indexes, keep_in_memory=True)

        if self.output_format == "list":
            rows: List[Batch] = [dict(row) for row in table]
            assert indexes[0] == rows[0][self.index_key], err_msg
        elif self.output_format == "dict":
            rows: Batch = table[None:None]
            assert indexes[0] == rows[self.index_key][0], err_msg
        else:
            raise ValueError(f"Unknown output format: {self.output_format}")

        # collate and return
        return self.collate_pipe(rows)
