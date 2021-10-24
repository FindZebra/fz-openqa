from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import dill
import rich
from datasets import Dataset
from rich.status import Status

from .base import Index
from .base import SearchResult
from fz_openqa.datamodules.index.utils.es_engine import ElasticSearchEngine
from fz_openqa.datamodules.pipes import Batchify
from fz_openqa.datamodules.pipes import CopyBatch
from fz_openqa.datamodules.pipes import DeBatchify
from fz_openqa.datamodules.pipes import MetaMapFilter
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import SciSpacyFilter
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import StopWordsFilter
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import get_separator


class ElasticSearchIndex(Index):
    index_name: Optional[str] = None
    preprocesing_pipe: Optional[Pipe] = None

    def __init__(
        self,
        *,
        index_key: str,
        text_key: str,
        query_key: str,
        batch_size: int = 32,
        num_proc: int = 1,
        filter_mode: Optional[str] = None,
        es: Optional[ElasticSearchEngine] = None,
        text_cleaner: Optional[TextFormatter] = TextFormatter(lowercase=True),
        **kwargs,
    ):
        super(ElasticSearchIndex, self).__init__(**kwargs)
        self.index_key = index_key
        self.text_key = text_key
        self.query_key = query_key
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.engine = es or ElasticSearchEngine()

        # text cleaning
        if isinstance(text_cleaner, TextFormatter):
            text_cleaner = text_cleaner.copy(text_key=self.text_key)
        self.text_cleaner = text_cleaner

        # pipe used to potentially filter the input text
        if filter_mode is not None:
            filter_pipe_cls = {
                "scispacy": SciSpacyFilter,
                "metamap": MetaMapFilter,
                "stopwords": StopWordsFilter,
            }[filter_mode]
            filter_pipe = filter_pipe_cls(text_key=self.text_key)
        else:
            filter_pipe = None

        # text cleaning and filtering
        self.preprocesing_pipe = Sequential(
            CopyBatch(),
            filter_pipe,
            text_cleaner,
        )

    def dill_inspect(self) -> Dict[str, Any]:
        return {
            "__all__": dill.pickles(self),
            "engine": dill.pickles(self.engine),
            "preprocesing_pipe": dill.pickles(self.preprocesing_pipe),
        }

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
        dataset = self.preprocess_text(dataset)

        # init the index
        self.index_name = dataset._fingerprint
        is_new_index = self.engine.es_create_index(self.index_name)

        # build the index
        if is_new_index:
            with Status("ingesting ES index.."):
                try:
                    response = self.engine.es_bulk(
                        index_name=self.index_name,
                        # todo: find a way to extract document titles
                        title="__no_title__",
                        document_idx=dataset[self.index_key],
                        document_txt=dataset[self.text_key],
                    )
                except Exception as ex:
                    # todo: catch a more precise exception
                    self.engine.es_remove_index(self.index_name)
                    raise ex

        if verbose:
            print(get_separator("="))
            print("=== build_es_index response ===")
            print(get_separator())
            rich.print(response)
            print(get_separator("="))

        self.is_indexed = True

    def search(self, query: Batch, k: int = 1, **kwargs) -> SearchResult:
        """
        Search the ES index for q batch of examples (query).

        Filter the incoming batch using the same pipe as the one
        used to build the index."""
        query = self.preprocesing_pipe(query, text_key=self.query_key)

        scores, indexes = self.engine.es_search_bulk(
            self.index_name, query[self.query_key], k=k
        )
        return SearchResult(score=scores, index=indexes)

    def search_one(
        self, query: Dict[str, Any], *, field: str = None, k: int = 1, **kwargs
    ) -> Tuple[List[float], List[int]]:
        """Search the index using the elastic search index for a single example."""
        query = Batchify()(query)
        query = self.preprocesing_pipe(query, text_key=self.query_key)
        query = DeBatchify()(query)

        results = self.engine.es_search(
            index_name=self.index_name,
            query=query[field],
            results=k,
        )

        scores = [eg["_score"] for eg in results["hits"]]
        indexes = [eg["_source"]["idx"] for eg in results["hits"]]
        return scores, indexes

    def preprocess_text(self, dataset: Dataset) -> Dataset:

        # process the dataset using the filtering pipe
        return dataset.map(
            self.preprocesing_pipe,
            batched=True,
            # @idariis: we need to decide how to set this, it depends on
            # the scispacy models
            # batch_size=self.batch_size,
            # @idariis: potentially increase this to `self.num_proc` to use multiprocessing
            num_proc=self.num_proc,
            desc="ES Indexing: preprocessing",
        )
