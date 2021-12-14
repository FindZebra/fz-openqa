from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

from datasets import Dataset
from omegaconf import OmegaConf
from rich.status import Status

from ..utils.dataset import keep_only_columns
from .base import Index
from .base import IndexMode
from .search_result import SearchResult
from fz_openqa.configs.datamodule.index_builder import es_body
from fz_openqa.datamodules.index.utils.es_engine import ElasticSearchEngine
from fz_openqa.datamodules.pipes import CopyBatch
from fz_openqa.datamodules.pipes import MetaMapFilter
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import SciSpacyFilter
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import StopWordsFilter
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.utils.datastruct import Batch

# load the default es configuration

DEFAULT_ES_BODY = OmegaConf.to_object(
    OmegaConf.load(Path(es_body.__file__).parent / "default.yaml")
)

logger = logging.getLogger("fz_openqa.index.elasticsearch")
DEFAULT_MAP_KWARGS = {"batch_size": 100, "num_proc": 4}


class ElasticSearchIndex(Index):
    preprocessing_pipe: Optional[Pipe] = None
    default_key: str = "text"
    no_fingerprint: List[str] = Index.no_fingerprint + ["engine", "map_kwargs"]

    def _prepare_index(
        self,
        *,
        filter_mode: Optional[str] = None,
        text_formatter: Optional[TextFormatter] = None,
        es_body: Optional[Dict] = DEFAULT_ES_BODY,
        analyze_es_tokens: Optional[bool] = False,
        map_kwargs: Optional[Dict] = None,
        **kwargs,
    ):

        if map_kwargs is None:
            map_kwargs = DEFAULT_MAP_KWARGS

        # set indexing parameters
        self.index_key = f"{self.index_field}.{self.index_output_key}"  # ie. "document.row_idx"
        self.map_kwargs = map_kwargs
        self.engine = ElasticSearchEngine(analyze=analyze_es_tokens)
        self.es_body = es_body
        self.analyze_es_tokens = analyze_es_tokens

        # input keys
        msg = f"ElasticSearch Index is only implemented for one key. Got {self.required_keys}"
        assert len(self.required_keys) == 1, msg
        self.index_text_key = self.input_keys(mode=IndexMode.INDEX)[0]
        self.query_text_key = self.input_keys(mode=IndexMode.QUERY)[0]
        self.text_keys = [self.index_text_key, self.query_text_key]

        # pipe used to potentially filter the input text
        if filter_mode is not None:
            filter_pipe_cls = {
                "scispacy": SciSpacyFilter,
                "metamap": MetaMapFilter,
                "stopwords": StopWordsFilter,
            }[filter_mode]
            filter_pipe = filter_pipe_cls(text_key=self.text_keys)
        else:
            filter_pipe = None

        # text cleaning and filtering ()
        if filter_pipe is not None or text_formatter is not None:
            if text_formatter is not None:
                text_formatter = text_formatter.copy(text_key=self.text_keys)
            self.preprocessing_pipe = Sequential(
                CopyBatch(),
                filter_pipe,
                text_formatter,
            )
        else:
            self.preprocessing_pipe = None

    def build(self, dataset: Dataset, verbose: bool = False, **kwargs):
        """Index the dataset using elastic search.
        We make sure a unique index is created for each dataset"""
        dataset = keep_only_columns(dataset, [self.index_key, self.index_text_key])
        dataset = self.preprocess_text(dataset)

        # set a unique index name
        self._set_index_name(dataset=dataset)

        # init the index
        is_new_index = self.engine.es_create_index(self.index_name, body=self.es_body)

        # build the index
        if is_new_index:
            with Status("ingesting ES index.."):
                try:
                    _ = self.engine.es_bulk(
                        index_name=self.index_name,
                        document_idx=dataset[self.index_key],
                        document_txt=dataset[self.index_text_key],
                    )
                except Exception as ex:
                    # clean up the index if something went wrong
                    self.engine.es_remove_index(self.index_name)
                    raise ex

        self.is_indexed = True

    def _preprocess_batch(self, batch: Batch, **kwargs) -> Batch:
        """Preprocess the batch before indexing"""
        if self.preprocessing_pipe is not None:
            batch = self.preprocessing_pipe(batch)
        return batch

    def _preprocess_query(self, batch: Batch, **kwargs) -> Batch:
        """Preprocess the batch before query"""
        if self.preprocessing_pipe is not None:
            batch = self.preprocessing_pipe(batch)
        return batch

    def search(self, query: Batch, *, k: int, **kwargs) -> SearchResult:
        """Search the index for the given query."""

        # fetch the texts
        texts = query[self.query_text_key]

        # query Elastic Search
        scores, indexes, contents = self.engine.es_search_bulk(self.index_name, texts, k=k)

        # if analyze is True, we need to fetch the analyzed text
        if self.analyze_es_tokens:
            analyzed_tokens = self.engine.es_analyze_text(self.index_name, contents)
        else:
            analyzed_tokens = None

        # build the results
        return SearchResult(
            score=scores,
            index=indexes,
            tokens=analyzed_tokens,
            dataset_size=self.dataset_size,
            k=k,
        )

    def preprocess_text(self, dataset: Dataset) -> Dataset:

        # process the dataset using the filtering pipe
        if self.preprocessing_pipe is None:
            return dataset

        return dataset.map(
            self.preprocessing_pipe,
            batched=True,
            **self.map_kwargs,
            desc="ES Indexing: preprocessing",
        )
