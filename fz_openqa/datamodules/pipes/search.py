from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import dill
import rich
import torch

from . import Collate
from . import Nest
from .base import Pipe
from .base import Rename
from fz_openqa.utils.datastruct import Batch


class SearchCorpus(Pipe):
    """Search a Corpus object given a query"""

    skip_fingerprint = True

    def __init__(
        self,
        corpus,
        *,
        k: Optional[int] = None,
        model: Optional[Union[Callable, torch.nn.Module]] = None,
        simple_collate: bool = False,
    ):
        assert corpus is not None, "corpus must be set."
        self.index = corpus._index
        self.dataset = corpus.dataset
        msg = (
            "Corpus.dataset is None, you probably need to "
            "`run corpus.setup()` before initializing this pipe"
        )
        assert self.dataset is not None, msg
        self.collate_pipe = corpus.collate_pipe
        self.k = k
        self.simple_collate = simple_collate
        self.model = model

    def __repr__(self):
        return self.as_fingerprintable().__repr__()

    def dill_inspect(self) -> Dict[str, Any]:
        return {
            "__all__": dill.pickles(self),
            "index": self.index.dill_inspect(),
            "dataset": dill.pickles(self.dataset),
            "collate_pipe": dill.pickles(self.collate_pipe),
        }

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "__all__": self._fingerprint(self),
            "index": self._fingerprint(self.index),
            "dataset": self._fingerprint(self.dataset),
            "collate_pipe": self._fingerprint(self.collate_pipe),
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
        simple_collate = simple_collate or self.simple_collate
        model = model or self.model

        # query the index
        search_result = self.index.search(
            query=query, k=k, model=model, **kwargs
        )

        # retrieve the examples from the dataset (flat list)
        flat_indexes = (idx for sub in search_result.index for idx in sub)
        flat_scores = (score for sub in search_result.score for score in sub)
        retrieved_docs = [
            {**self.dataset[idx], "document.retrieval_score": score}
            for idx, score in zip(flat_indexes, flat_scores)
        ]
        if simple_collate:
            flat_docs_batch = Collate()(retrieved_docs)
        else:
            flat_docs_batch = self.collate_pipe(retrieved_docs)

        # nest the examples:
        # [eg for eg in examples] -> [[eg_q for eg_q in results[q] for q in query]
        return Nest(stride=k)(flat_docs_batch)
