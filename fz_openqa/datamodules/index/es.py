from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import rich
from datasets import Dataset

from ...utils.datastruct import Batch
from ...utils.es_functions import ElasticSearch
from ...utils.pretty import get_separator
from ..pipes import MetaMapFilter
from ..pipes import Pipe
from ..pipes import PrintBatch
from ..pipes import SciSpacyFilter
from ..pipes import Sequential
from ..pipes import StopWordsFilter
from .base import Index
from .base import SearchResult


class ElasticSearchIndex(Index):
    index_name: Optional[str] = None
    filter_pipe: Optional[Pipe] = None

    def __init__(
        self,
        index_key: str,
        text_key: str,
        query_key: str,
        batch_size: int = 32,
        filter_mode: Optional[str] = None,
        es: Optional[ElasticSearch] = None,
        **kwargs,
    ):
        super(ElasticSearchIndex, self).__init__(**kwargs)
        self.index_key = index_key
        self.text_key = text_key
        self.query_key = query_key
        self.batch_size = batch_size
        self.engine = es or ElasticSearch()

        # pipe used to potentially filter the input text
        if filter_mode is not None:
            filter_pipe_cls = {
                "scispacy": SciSpacyFilter,
                "metamap": MetaMapFilter,
                # @vlievin: fyi has to be included in the datamodule
                "stopwords": StopWordsFilter,
            }[filter_mode]

            self.filter_pipe = Sequential(
                # @idariis: added this line for debugging
                PrintBatch(header="filtering input"),
                filter_pipe_cls(text_key=self.text_key),
                # @filter_pipe_cls: added this line for debugging
                PrintBatch(header="filtering output"),
            )

    def build(self, dataset: Dataset, verbose: bool = False, **kwargs):
        """Index the dataset using elastic search.
        We make sure a unique index is created for each dataset"""

        # preprocess the dataset
        unused_columns = [
            c
            for c in dataset.column_names
            if c not in [self.index_key, self.text_key]
        ]
        dataset = dataset.remove_columns(unused_columns)
        dataset = self.filter_text(dataset)

        # init the index
        self.index_name = dataset._fingerprint
        is_new_index = self.engine.es_create_index(self.index_name)

        # build the index
        if is_new_index:
            _ = self.engine.es_bulk(
                index_name=self.index_name,
                # todo: find a way to extract document titles
                title="__no_title__",
                document_idx=dataset[self.index_key],
                document_txt=dataset[self.text_key],
            )

        if verbose:
            print(get_separator("="))
            print("=== build_es_index response ===")
            print(get_separator())
            rich.print(response)
            print(get_separator("="))

        self.is_indexed = True

    def search(
        self, query: Batch, k: int = 1, **kwargs
    ) -> SearchResult:
        """filter the incoming batch using the same pipe as the one
        used to build the index."""
        if self.filter_pipe is not None:
            query = self.filter_pipe(query, text_key=self.query_key)

        scores, indexes = self.engine.es_search_bulk(
            self.index_name, 
            query[self.query_key], 
            k=k
        )

        # todo:  check that this works, not sure if that does the trick
        # make sure that "indexes" correspond to the global index field "idx",
        # so we can use the index values to index the dataset object: (i.e. `corpus.dataset[idx]`)
        return SearchResult(score=scores, index=indexes)

    def search_one(
        self, query: Dict[str, Any], *, field: str = None, k: int = 1, **kwargs
    ) -> Tuple[List[float], List[int]]:
        """Search the index using the elastic search index"""

        results = self.engine.es_search(
            index_name=self.index_name,
            query=query[field],
            results=k,
        )

        scores = [eg["_score"] for eg in results["hits"]]
        indexes = [eg["_source"]["idx"] for eg in results["hits"]]
        return scores, indexes

    def filter_text(self, dataset: Dataset) -> Dataset:
        if self.filter_pipe is None:
            return dataset

        # process the dataset using the filtering pipe
        return dataset.map(
            self.filter_pipe,
            batched=True,
            # @idariis: we need to decide how to set this
            batch_size=self.batch_size,
            # @idariis: potentially increase this to `self.num_proc` to use multiprocessing
            num_proc=1,
            desc="Computing corpus vectors",
        )
