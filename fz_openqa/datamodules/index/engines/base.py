from __future__ import annotations

import abc
import json
from copy import copy
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import torch
from datasets import Dataset
from datasets.search import SearchResults
from hydra.utils import instantiate
from loguru import logger

from fz_openqa.datamodules.index._base import camel_to_snake
from fz_openqa.datamodules.index._base import slice_batch
from fz_openqa.datamodules.index.engines.vector_base.utils.faiss import Tensors
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.array import FormatArray
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import OutputFormat
from fz_openqa.utils.datastruct import PathLike
from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.utils.functional import infer_batch_size
from fz_openqa.utils.pretty import pprint_batch
from fz_openqa.utils.tensor_arrow import TensorArrowTable


class IndexEngine(Pipe, metaclass=abc.ABCMeta):
    """This class implements an index."""

    no_fingerprint: List[str] = ["k", "path", "max_batch_size"]
    no_index_name: List[str] = []
    index_columns: List[str] = []
    query_columns: List[str] = []
    _default_config: Dict[str, Any] = {}
    output_score_key = "document.proposal_score"
    output_index_key = "document.row_idx"

    def __init__(
        self,
        *,
        path: PathLike,
        k: int = 10,
        max_batch_size: Optional[int] = None,
        # Pipe args
        input_filter: None = None,
        update: bool = False,
        # arguments registered in `config`
        config: Dict[str, Any] = None,
    ):
        super().__init__(input_filter=input_filter, update=update)
        # default number of retrieved documents
        self.k = k
        self.max_batch_size = max_batch_size

        # set the index configuration
        self.config = self.default_config
        if config is not None:
            self.config.update(config)

        # set the path where to solve the configuration and data
        self.path = None if path is None else Path(path)

    @classmethod
    def load_from_path(cls, path: PathLike):
        state_path = Path(path) / "state.json"
        with open(state_path, "r") as f:
            config = json.load(f)
            instance = instantiate(config)
            instance.load()

        return instance

    def save(self):
        """save the index to file"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.state_file), "w") as f:
            f.write(json.dumps(self._get_state()))

    def load(self):
        """save the index to file"""
        with open(str(self.state_file), "r") as f:
            state = json.load(f)
            self.k = state.pop("k")
            self.max_batch_size = state.pop("max_batch_size")
            self.config = state["config"]

    @property
    def state_file(self) -> Path:
        return Path(self.path) / "state.json"

    @property
    def default_config(self):
        return copy(self._default_config)

    def _get_state(self) -> Dict[str, Any]:
        state = {}
        state["config"] = copy(self.config)
        state["path"] = str(self.path)
        state["k"] = self.k
        state["max_batch_size"] = self.max_batch_size
        state["_target_"] = type(self).__module__ + "." + type(self).__qualname__
        return state

    def build(
        self,
        *,
        vectors: Optional[Tensors | TensorArrowTable] = None,
        corpus: Optional[Dataset] = None,
    ):
        if self.exists():
            logger.info(f"Loading index from {self.path}")
            self.load()
        else:
            logger.info(f"Creating index at {self.path}")
            self._build(vectors=vectors, corpus=corpus)
            self.save()
            assert self.exists(), f"Index {type(self).__name__} was not created."

    def rm(self):
        """Remove the index."""
        if self.path.exists():
            if self.path.is_dir():
                self.path.rmdir()
            else:
                self.path.unlink()

    def exists(self):
        """Check if the index exists."""
        return self.path.exists()

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        ...

    @abc.abstractmethod
    def _build(self, *data: Any, **kwargs):
        """build the index from the vectors or text."""
        ...

    @abc.abstractmethod
    def cpu(self):
        """Move the index to CPU."""
        ...

    @abc.abstractmethod
    def cuda(self, devices: Optional[List[int]] = None):
        """Move the index to CUDA."""
        ...

    @abc.abstractmethod
    def free_memory(self):
        """Free the memory of the index."""
        ...

    @property
    @abc.abstractmethod
    def is_up(self) -> bool:
        """Check if the index is up."""
        ...

    @abc.abstractmethod
    def search(self, *query: Any, k: int = None, **kwargs) -> SearchResults:
        ...

    def _get_index_name(self, dataset, config) -> str:
        """Set the index name. Must be unique to allow for sage caching."""
        cls_id = camel_to_snake(type(self).__name__)
        dcfg = self.deterministic_config(config)
        cfg_id = get_fingerprint(dcfg)
        return f"{cls_id}-{dataset._fingerprint}-{cfg_id}"

    def deterministic_config(self, config):
        config = {key: v for key, v in config.items() if key not in self.no_index_name}
        return config

    def fingerprint(
        self, reduce=False, exclude: Optional[List[str]] = None
    ) -> str | Dict[str, Any]:
        """
        Return a fingerprint of the object. All attributes stated in `no_fingerprint` are excluded.
        """

        fingerprints = self._get_fingerprint_struct()

        # clean the config object]
        fingerprints["config"] = {
            key: v for key, v in fingerprints["config"].items() if key not in self.no_fingerprint
        }

        if reduce:
            fingerprints = get_fingerprint(fingerprints)

        return fingerprints

    def _call_batch(
        self,
        query: Batch,
        idx: Optional[List[int]] = None,
        k: Optional[int] = None,
        vectors: Optional[Tensors] = None,
        output_format: Optional[OutputFormat] = None,
        **kwargs,
    ) -> Batch:
        """
        Search the index for a batch of examples (query).

        Filter the incoming batch using the same pipe as the one
        used to build the index."""
        k = k or self.k

        # fetch vectors
        if vectors is None:
            q_vectors = None
        else:
            q_vectors = vectors[idx]

        # search the index by chunk
        batch_size = infer_batch_size(query)
        search_results = None
        if self.max_batch_size is not None:
            eff_batch_size = min(max(1, self.max_batch_size), batch_size)
        else:
            eff_batch_size = batch_size
        for i in range(0, batch_size, eff_batch_size):

            # slice the query
            chunk_i = slice_batch(query, slice(i, i + eff_batch_size))
            if q_vectors is not None:
                q_vectors_i = q_vectors[i : i + eff_batch_size]
            else:
                q_vectors_i = None

            # search
            r = self._search_chunk(chunk_i, k=k, vectors=q_vectors_i, **kwargs)

            if search_results is None:
                search_results = r
            else:
                search_results += r

        # format the output
        formatter = FormatArray(output_format)
        output = {
            self.output_index_key: formatter(search_results.index),
            self.output_score_key: formatter(search_results.score),
        }

        pprint_batch(output, "Engine out")

        return output

    @abc.abstractmethod
    def _search_chunk(
        self, query: Batch, *, k: int, vectors: Optional[torch.Tensor], **kwargs
    ) -> SearchResult:
        raise NotImplementedError

    def __del__(self):
        self.free_memory()
