from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import rich
from datasets import Dataset
from elasticsearch import Elasticsearch
from loguru import logger
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from fz_openqa.configs.datamodule.index_builder import es_body
from fz_openqa.datamodules.index.engines.base import IndexEngine
from fz_openqa.datamodules.index.engines.utils.es import es_create_index
from fz_openqa.datamodules.index.engines.utils.es import es_ingest_bulk
from fz_openqa.datamodules.index.engines.utils.es import es_remove_index
from fz_openqa.datamodules.index.engines.utils.es import es_search_bulk
from fz_openqa.datamodules.index.engines.vector_base.utils.faiss import TensorLike
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.datamodules.utils.dataset import keep_only_columns
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import pprint_batch
from fz_openqa.utils.tensor_arrow import TensorArrowTable

DEFAULT_ES_BODY = OmegaConf.to_object(
    OmegaConf.load(Path(es_body.__file__).parent / "default.yaml")
)


def ping_es() -> bool:
    return Elasticsearch().ping()


class ElasticsearchEngine(IndexEngine):
    """This class an Elastic Search index."""

    _instance: Elasticsearch
    index_columns: List[str] = ["document.row_idx", "document.text"]
    query_columns: List[str] = ["question.text", "answer.text"]
    no_fingerprint: List[str] = IndexEngine.no_fingerprint + [
        "_instance",
        "timeout",
        "es_logging_level",
        "ingest_batch_size",
    ]
    no_index_name = IndexEngine.no_index_name + [
        "_instance" "timeout",
        "es_logging_level",
        "auxiliary_weight",
        "es_temperature",
        "ingest_batch_size",
    ]
    _default_config: Dict[str, Any] = {
        "timeout": 60,
        "es_body": DEFAULT_ES_BODY,
        "index_key": "document.row_idx",
        "index_text_key": "document.text",
        "ingest_batch_size": 1000,
        "auxiliary_weight": 0,
        "es_temperature": 1.0,
        "es_logging_level": "error",
    }

    def _build(
        self,
        vectors: Optional[TensorLike | TensorArrowTable] = None,
        corpus: Optional[Dataset] = None,
    ):

        if corpus is None:
            raise ValueError("The corpus is required.")

        # keep only the relevant columns
        corpus = keep_only_columns(corpus, self.index_columns)

        # set a unique index name
        self.config["index_name"] = self._get_index_name(corpus, self.config)
        self.config["corpus_size"] = len(corpus)

        # instantiate the ElasticSearch instance
        self._init_es_instance()

        # init the index
        is_new_index = es_create_index(
            self.instance, self.config["index_name"], body=self.config["es_body"]
        )
        if not is_new_index:
            logger.info(
                f"ElasticSearch index with name=`{self.config['index_name']}` already exists."
            )

        # build the index
        if is_new_index:
            batch_size = self.config["ingest_batch_size"]
            try:
                for i in tqdm(range(0, len(corpus), batch_size), desc="Ingesting ES index"):
                    batch = corpus[i : i + batch_size]
                    _ = es_ingest_bulk(
                        self.instance,
                        index_name=self.config["index_name"],
                        document_idx=batch[self.config["index_key"]],
                        document_txt=batch[self.config["index_text_key"]],
                    )
            except Exception as ex:
                # clean up the index if something went wrong
                es_remove_index(self.instance, self.config["index_name"])
                raise ex

    def _init_es_instance(self):
        log_level = {
            "error": logging.ERROR,
            "warning": logging.WARNING,
            "info": logging.INFO,
            "debug": logging.DEBUG,
        }[self.config["es_logging_level"]]
        logging.getLogger("elasticsearch").setLevel(log_level)
        self._instance = Elasticsearch(timeout=self.config["timeout"])

    def load(self):
        """save the index to file"""
        super().load()
        self._init_es_instance()

    @property
    def instance(self):
        return self._instance

    def rm(self):
        """Remove the index."""
        es_remove_index(self.instance, self.config["index_name"])

    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        return -1

    def cpu(self):
        """Move the index to CPU."""
        pass

    def cuda(self, devices: Optional[List[int]] = None):
        """Move the index to CUDA."""
        pass

    def free_memory(self):
        """Free the memory of the index."""
        pass

    @property
    def is_up(self) -> bool:
        """Check if the index is up."""
        is_new_index = es_create_index(self.instance, self.config["index_name"])
        istance_init = hasattr(self, "_instance")
        return istance_init is not None and not is_new_index

    def search(
        self, *queries: (List[str], Optional[List[str]]), k: int = None, **kwargs
    ) -> SearchResult:
        """Search the index for a query and return the top-k results."""
        k = k or self.k

        # unpack args
        query_text, auxiliary_text = queries
        config = self.config

        # query Elastic Search
        scores, indexes, contents = es_search_bulk(
            self.instance,
            index_name=config["index_name"],
            queries=query_text,
            k=k,
            auxiliary_queries=auxiliary_text,
            auxiliary_weight=config["auxiliary_weight"],
        )

        if config["es_temperature"] is not None:
            scores = [[s / config["es_temperature"] for s in s_list] for s_list in scores]

        # build the results
        return SearchResult(
            score=scores,
            index=indexes,
            dataset_size=config["corpus_size"],
            k=k,
        )

    def _search_chunk(
        self, batch: Batch, idx: Optional[List[int]] = None, **kwargs
    ) -> SearchResult:

        args = [batch.get(key, None) for key in self.query_columns]

        # check the arguments
        if self.config["auxiliary_weight"] > 0 and args[1] is None:
            raise ValueError(
                f"Missing auxiliary text "
                f"(column={self.query_columns[1]}) required "
                f"for auxiliary weight > 0"
            )

        search_result = self.search(*args, **kwargs)

        return search_result

    def __getstate__(self):
        """this method is called when attempting pickling.
        ES instances cannot be properly pickled"""
        state = self.__dict__.copy()
        # Don't pickle the ES instances
        for attr in ["_instance"]:
            if attr in state:
                state.pop(attr)

        return state

    def __setstate__(self, state):
        self.__dict__ = state
        state["_instance"] = Elasticsearch(timeout=state["config"]["timeout"])
