from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import rich
from datasets import Dataset
from omegaconf import OmegaConf
from rich.status import Status

from .base import Index
from .base import IndexMode
from .search_result import SearchResult
from fz_openqa.configs.datamodule.index_builder import es_body
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

# load the default es configuration
DEFAULT_ES_BODY = OmegaConf.to_object(
    OmegaConf.load(Path(es_body.__file__).parent / "default.yaml")
)


class ElasticSearchIndex(Index):
    preprocesing_pipe: Optional[Pipe] = None

    def __init__(
        self,
        dataset: Dataset,
        *,
        index_key: str = "document.row_idx",
        required_keys: List[str] = None,
        query_field: str = "question",
        index_field: str = "document",
        index_name: Optional[str] = None,
        batch_size: int = 32,
        num_proc: int = 1,
        filter_mode: Optional[str] = None,
        text_cleaner: Optional[TextFormatter] = None,
        es_body: Optional[Dict] = DEFAULT_ES_BODY,
        analyze: Optional[bool] = False,
        **kwargs,
    ):
        if required_keys is None:
            required_keys = ["text"]

        self.index_field = index_field
        self.query_field = query_field
        self.required_keys = required_keys
        self.index_key = index_key
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.engine = ElasticSearchEngine(analyze=False)
        self.es_body = es_body
        self.analyze = analyze

        # input keys
        msg = f"ElasticSearch Index is only implemented for one key. Got {required_keys}"
        assert len(required_keys) == 1, msg
        self.index_text_key = self.input_keys(mode=IndexMode.INDEX)[0]
        self.query_text_key = self.input_keys(mode=IndexMode.QUERY)[0]

        # pipe used to potentially filter the input text
        if filter_mode is not None:
            filter_pipe_cls = {
                "scispacy": SciSpacyFilter,
                "metamap": MetaMapFilter,
                "stopwords": StopWordsFilter,
            }[filter_mode]
            filter_pipe = filter_pipe_cls(text_key=self.index_text_key)
        else:
            filter_pipe = None

        # text cleaning and filtering ()
        if filter_pipe is not None and text_cleaner is not None:
            self.preprocesing_pipe = Sequential(
                CopyBatch(),
                filter_pipe,
                text_cleaner,
            )
        else:
            self.preprocesing_pipe = None

        # call the super to index the dataset
        kwargs.update(
            required_keys=self.required_keys,
            query_field=self.query_field,
            index_field=self.index_field,
            index_name=index_name,
        )
        super(ElasticSearchIndex, self).__init__(dataset=dataset, **kwargs)

    def build(self, dataset: Dataset, verbose: bool = False, **kwargs):
        """Index the dataset using elastic search.
        We make sure a unique index is created for each dataset"""
        # preprocess the dataset
        cols_to_drop = [
            c for c in dataset.column_names if c not in [self.index_key, self.index_text_key]
        ]
        dataset = dataset.remove_columns(cols_to_drop)
        dataset = self.preprocess_text(dataset)

        # init the index
        self.index_name = Pipe._fingerprint(
            {
                "fingerprint": dataset._fingerprint,
                "es_body": self.es_body,
                "analyse": self.analyze,
            }
        )
        is_new_index = self.engine.es_create_index(self.index_name, body=self.es_body)

        # build the index
        if is_new_index:
            with Status("ingesting ES index.."):
                try:
                    response = self.engine.es_bulk(
                        index_name=self.index_name,
                        document_idx=dataset[self.index_key],
                        document_txt=dataset[self.index_text_key],
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
        if self.preprocesing_pipe is not None:
            query = self.preprocesing_pipe(query, text_key=self.query_text_key)

        scores, indexes, contents = self.engine.es_search_bulk(
            self.index_name, query[self.query_text_key], k=k
        )

        if self.analyze:
            analyzed_tokens = self.engine.es_analyze_text(self.index_name, contents)
        else:
            analyzed_tokens = None

        return SearchResult(
            score=scores,
            index=indexes,
            tokens=analyzed_tokens,
            dataset_size=self.dataset_size,
            k=k,
        )

    def search_one(
        self, query: Dict[str, Any], *, k: int = 1, **kwargs
    ) -> Tuple[List[float], List[int]]:
        """Search the index using the elastic search index for a single example."""
        query = Batchify()(query)
        query = self.preprocesing_pipe(query, text_key=self.query_text_key)
        query = DeBatchify()(query)

        results = self.engine.es_search(
            index_name=self.index_name,
            query=query[self.query_text_key],
            results=k,
        )

        scores = [eg["_score"] for eg in results["hits"]]
        indexes = [eg["_source"]["idx"] for eg in results["hits"]]
        return scores, indexes

    def preprocess_text(self, dataset: Dataset) -> Dataset:

        # process the dataset using the filtering pipe
        if self.preprocesing_pipe is None:
            return dataset

        return dataset.map(
            self.preprocesing_pipe,
            batched=True,
            num_proc=self.num_proc,
            desc="ES Indexing: preprocessing",
        )
