from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import dill
import rich
import torch
from datasets import Dataset

from . import Collate
from . import Nest
from .base import Pipe
from fz_openqa.utils.datastruct import Batch


class SearchCorpus(Pipe):
    """Search a Corpus object given a query"""

    def __init__(
        self,
        corpus,
        *,
        k: Optional[int] = None,
        model: Optional[Union[Callable, torch.nn.Module]] = None,
        index_output_key: str = "document.row_idx",
        score_output_key: str = "document.retrieval_score",
        **kwargs,
    ):
        super(SearchCorpus, self).__init__(**kwargs)
        assert corpus is not None, "corpus must be set."
        self.index = corpus._index
        self.dataset = corpus.dataset
        msg = (
            "Corpus.dataset is None, you probably need to "
            "`run corpus.setup()` before initializing this pipe"
        )
        assert self.dataset is not None, msg
        self.index_output_key = index_output_key
        self.score_output_key = score_output_key
        self.k = k
        self.model = model

    def __repr__(self):
        return self.as_fingerprintable().__repr__()

    def dill_inspect(self) -> Dict[str, Any]:
        return {
            "__all__": dill.pickles(self),
            "index": self.index.dill_inspect(),
            "dataset": dill.pickles(self.dataset),
        }

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "__all__": self._fingerprint(self),
            "index": self._fingerprint(self.index),
            "dataset": self._fingerprint(self.dataset),
        }

    def as_fingerprintable(self) -> Optional:
        return {
            "__type__": type(self),
            "k": self.k,
            "dataset": self.dataset._fingerprint,
            "es_index": self.index.index_name,
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
        dataset: Dataset,
        keys: Optional[List[str]] = None,
        collate_pipe: Pipe = Collate(),
        index_key: str = "document.row_idx",
        **kwargs,
    ):
        super(FeatchDocuments, self).__init__(**kwargs)
        self.dataset = dataset
        self.keys = keys
        self.collate_pipe = collate_pipe
        self.index_key = index_key

    def output_keys(self, input_keys: List[str]) -> List[str]:
        features = self.dataset.column_names
        return list(
            self._filter_row(
                {c: None for c in features}, keys=self.keys
            ).values()
        )

    @staticmethod
    def _filter_row(
        row: Dict[str, Any], *, keys: Optional[List[str]]
    ) -> Dict[str, Any]:
        return {k: v for k, v in row.items() if keys is None or k in keys}

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        # todo: check dataset fingerprint

        # get the `dataset` indexes
        indexes = (int(idx) for idx in batch[self.index_key])

        if self.keys is not None and len(self.keys) == 1:
            key = self.keys[0]
            # fetch documents
            retrieved_docs = map(self.dataset[key].__getitem__, indexes)
            retrieved_docs = [{key: x} for x in retrieved_docs]
        else:
            # fetch documents
            retrieved_docs = map(self.dataset.__getitem__, indexes)
            # filter keys
            retrieved_docs = list(
                self._filter_row(x, keys=self.keys) for x in retrieved_docs
            )
        # collate and return
        return self.collate_pipe(retrieved_docs)
