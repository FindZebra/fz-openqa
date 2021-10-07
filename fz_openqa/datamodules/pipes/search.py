from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import dill
import torch

from . import Collate
from . import Nest
from .base import Pipe
from .base import Rename
from fz_openqa.utils.datastruct import Batch


class SearchCorpus(Pipe):
    """Search a Corpus object given a query"""

    def __init__(
        self,
        corpus,
        *,
        k: Optional[int] = None,
        model: Optional[Union[Callable, torch.nn.Module]] = None,
        simple_collate: bool = False,
        update_query: bool = True,
    ):
        self.index = corpus._index
        self.dataset = corpus.dataset
        assert (
            self.dataset is not None
        ), "Corpus.dataset is None, you probably need to `run corpus.setup()` before initializing this pipe"
        self.collate_pipe = corpus.collate_pipe
        self.k = k
        self.update_query = update_query
        self.simple_collate = simple_collate
        self.model = model

    def dill_inspect(self) -> Dict[str, Any]:
        return {
            "__all__": dill.pickles(self),
            "index": self.index.dill_inspect(),
            "dataset": dill.pickles(self.dataset),
            "collate_pipe": dill.pickles(self.collate_pipe),
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
            flat_docs_batch = Collate(keys=None)(retrieved_docs)
        else:
            flat_docs_batch = self.collate_pipe(retrieved_docs)

        # nest the examples:
        # [eg for eg in examples] -> [[eg_q for eg_q in results[q] for q in query]
        output = Nest(stride=k)(flat_docs_batch)

        if self.update_query:
            output = Rename({"idx": "document.global_idx"})(output)
            query.update(output)
            return query

        else:
            return output
