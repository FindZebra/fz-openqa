from __future__ import annotations

import re
from abc import abstractmethod
from enum import Enum
from typing import List
from typing import Optional

import rich
from datasets import Dataset

from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.utils.array import FormatArray
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import OutputFormat
from fz_openqa.utils.functional import infer_batch_size


class IndexMode(Enum):
    INDEX = "index"
    QUERY = "query"


def slice_batch(batch: Batch, i: int | slice) -> Batch:
    return {k: v[i] for k, v in batch.items()}


def camel_to_snake(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


class Index(Pipe):
    """Keep an index of a Dataset and search it using queries."""

    index_name: Optional[str] = None
    is_indexed: bool = False
    default_key: Optional[str | List[str]] = None
    no_fingerprint: List[str] = ["verbose", "index_name", "max_chunksize", "id", "k"]

    def __init__(
        self,
        dataset: Dataset,
        *,
        k: int = 10,
        required_keys: str | List[str] = None,
        query_field: str = "question",
        index_field: str = "document",
        verbose: bool = False,
        max_chunksize: int = 100,
        index_output_key: str = "row_idx",
        score_output_key: str = "retrieval_score",
        analyzed_output_key: str = "analyzed_tokens",
        # `Pipe` arguments
        id: Optional[str] = None,
        update: bool = False,
        input_filter: None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        dataset
            The dataset to be index (corpus)
        k
            Default number of results to return for each query
        required_keys
            The keys required for each field. e.g. ['text']]
        query_field
            The field to be used as query (in the query dataset). e.g. question
        index_field
            The field to be used as index (in the dataset). e.g. document
        verbose
            Whether to print the progress
        max_chunksize
            The maximum size of a chunk to be queried
        kwargs
            Other parameters to be passed to the `build` method
        """
        if required_keys is None:
            required_keys = self.default_key
        if isinstance(required_keys, str):
            required_keys = [required_keys]

        # indexing parameters
        self.k = k
        self.max_chunksize = max_chunksize

        # input fields & keys
        self.query_field = query_field
        self.index_field = index_field
        self.required_keys = required_keys

        # meta parameters
        self.verbose = verbose
        self.dataset_size = len(dataset)

        # output keys
        self.index_output_key = index_output_key
        self.score_output_key = score_output_key
        self.analyzed_output_key = analyzed_output_key

        # call the Pipe __init__
        assert input_filter is None, "input_filter is set automatically, please do not set it"
        input_filter = In(self.input_keys(IndexMode.QUERY))
        super(Index, self).__init__(id=id, update=update, input_filter=input_filter)

        # prepare index, such as defining the preprocessing ops
        self._prepare_index(**kwargs)

        # build the index
        self.build(dataset=dataset, **kwargs)

    def _prepare_index(self, **kwargs):
        """prepare index, such as defining the preprocessing ops"""
        pass

    @abstractmethod
    def build(self, dataset: Dataset, **kwargs):
        """Index a dataset."""
        raise NotImplementedError

    @abstractmethod
    def _set_index_name(self, dataset) -> None:
        """Set the index name. Must be unique to allow for sage caching."""
        cls_id = camel_to_snake(type(self).__name__)
        pipe_fingerprint = self.fingerprint(reduce=True)
        self.index_name = f"{cls_id}-{dataset._fingerprint}-{pipe_fingerprint}"

    @abstractmethod
    def _search_chunk(self, query: Batch, *, k: int, **kwargs) -> SearchResult:
        raise NotImplementedError

    def _preprocess_query(self, batch: Batch, **kwargs) -> Batch:
        """Preprocess the batch before query"""
        return batch

    def input_keys(self, mode: Optional[IndexMode] = None) -> List[str]:
        """Return the list of keys required for each mode (indexing, querying)"""
        output = []
        if mode is None or mode == IndexMode.INDEX:
            output += [f"{self.index_field}.{key}" for key in self.required_keys]

        if mode is None or mode == IndexMode.QUERY:
            output += [f"{self.query_field}.{key}" for key in self.required_keys]

        return output

    def __repr__(self):
        params = {"is_indexed": self.is_indexed, "index_name": self.index_name}
        params = [f"{k}={v}" for k, v in params.items()]
        return f"{self.__class__.__name__}({', '.join(params)})"

    def _call_batch(
        self,
        query: Batch,
        idx: Optional[List[int]] = None,
        k: Optional[int] = None,
        output_format: Optional[OutputFormat] = None,
        **kwargs,
    ) -> Batch:
        """
        Search the index for a batch of examples (query).

        Filter the incoming batch using the same pipe as the one
        used to build the index."""
        k = k or self.k

        query = self._preprocess_query(query, idx=idx, **kwargs)
        if len(query.keys()) == 0:
            raise ValueError(
                f"No query keys found in batch {query}, "
                f"make sure the input batch has the following "
                f"keys={self.input_keys(IndexMode.QUERY)}"
            )

        # search the index by chunk
        batch_size = infer_batch_size(query)
        search_results = None
        eff_batch_size = min(max(1, self.max_chunksize), batch_size)
        for i in range(0, batch_size, eff_batch_size):
            chunk_i = slice_batch(query, slice(i, i + eff_batch_size))
            r = self._search_chunk(chunk_i, k=k, **kwargs)

            if search_results is None:
                search_results = r
            else:
                search_results += r

        return self._format_output(search_results, output_format=output_format)

    def search(self, *args, **kwargs) -> Batch:
        return self.__call__(*args, **kwargs)

    def _format_output(
        self, search_results: SearchResult, output_format: Optional[OutputFormat] = None
    ) -> Batch:
        """Format the output of the search"""

        output = {
            self.index_output_key: search_results.index,
            self.score_output_key: search_results.score,
        }

        if search_results.tokens is not None:
            output[self.analyzed_output_key] = search_results.tokens

        output = {f"{self.index_field}.{key}": value for key, value in output.items()}

        if output_format is not None:
            formatter = FormatArray(output_format=output_format)
            output = {k: formatter(v) for k, v in output.items()}

        return output

    def to_cpu(self):
        pass
