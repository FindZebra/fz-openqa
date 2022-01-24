from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import rich
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
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import TextFilter
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
    no_fingerprint: List[str] = Index.no_fingerprint + ["engine", "prep_map_kwargs"]
    _es_chunk_size: int = 10

    def _prepare_index(
        self,
        *,
        text_filter: Optional[TextFilter] = None,
        text_formatter: Optional[TextFormatter] = None,
        es_body: Optional[Dict] = DEFAULT_ES_BODY,
        analyze_es_tokens: Optional[bool] = False,
        prep_map_kwargs: Optional[Dict] = None,
        es_temperature: Optional[float] = 1.0,
        **kwargs,
    ):

        if prep_map_kwargs is None:
            prep_map_kwargs = DEFAULT_MAP_KWARGS

        # set indexing parameters
        self.index_key = f"{self.index_field}.{self.index_output_key}"  # ie. "document.row_idx"
        self.prep_map_kwargs = prep_map_kwargs
        self.engine = ElasticSearchEngine(analyze=analyze_es_tokens)
        self.es_body = es_body
        self.analyze_es_tokens = analyze_es_tokens
        self.temperature = es_temperature

        # override max_chunk_size
        if self.max_chunksize != self._es_chunk_size:
            warnings.warn(f"max_chunksize is set to {self.max_chunksize}")
        self.max_chunksize = self._es_chunk_size

        # input keys
        msg = f"ElasticSearch Index is only implemented for one key. Got {self.required_keys}"
        assert len(self.required_keys) == 1, msg
        self.index_text_key = self.input_keys(mode=IndexMode.INDEX)[0]
        self.query_text_key = self.input_keys(mode=IndexMode.QUERY)[0]
        self.text_keys = [self.index_text_key, self.query_text_key]

        # pipe used to potentially filter the input text
        if text_filter is not None:
            text_filter: TextFilter = text_filter.copy(text_key=self.text_keys)
            logger.info(f"Using {text_filter} to filter text")

        if text_formatter is not None:
            text_formatter: TextFormatter = text_formatter.copy(text_key=self.text_keys)
            logger.info(f"Using {text_formatter} to clean text")

        # text cleaning and filtering
        if text_filter is not None or text_formatter is not None:
            self.preprocessing_pipe = Sequential(
                CopyBatch(),
                text_filter,
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
                # todo: to it by batch and add progress bar. This is currently loading
                #  ALL text into memory
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

    def _preprocess_query(self, batch: Batch, **kwargs) -> Batch:
        """Preprocess the batch before query"""
        if self.preprocessing_pipe is not None:
            batch = self.preprocessing_pipe(batch)
        return batch

    def _search_chunk(self, query: Batch, *, k: int, **kwargs) -> SearchResult:
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

        if self.temperature is not None:
            scores = [[s / self.temperature for s in s_list] for s_list in scores]

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
            **self.prep_map_kwargs,
            desc="ES Indexing: preprocessing",
        )
