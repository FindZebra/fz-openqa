from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import rich
import torch  # type: ignore
from datasets import Dataset
from datasets import Split
from datasets.search import SearchResults
from loguru import logger
from pytorch_lightning import Trainer

from fz_openqa.datamodules.index.engines.base import IndexEngine
from fz_openqa.datamodules.index.engines.vector_base import VectorBase
from fz_openqa.datamodules.index.engines.vector_base.auto import AutoVectorBase
from fz_openqa.datamodules.index.engines.vector_base.utils.faiss import Tensors
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import OutputFormat
from fz_openqa.utils.metric_type import MetricType
from fz_openqa.utils.pretty import pprint_batch
from fz_openqa.utils.tensor_arrow import TensorArrowTable


class FaissEngine(IndexEngine):
    """This class implements a low level index."""

    _max_num_proc: int = 1

    index_columns: List[str] = ["_hd_"]
    query_columns: List[str] = ["_hq_"]
    no_fingerprint: List[str] = IndexEngine.no_fingerprint + [
        "loader_kwargs",
        "keep_on_cpu",
        "train_on_cpu",
        "_index",
        "tempmem",
    ]
    no_index_name = IndexEngine.no_index_name + [
        "_instance" "timeout",
        "es_logging_level",
        "auxiliary_weight",
        "es_temperature",
        "ingest_batch_size",
    ]

    _default_config: Dict[str, Any] = {
        "index_factory": "Flat",
        "nprobe": 32,
        "keep_on_cpu": False,
        "train_on_cpu": False,
        "train_size": 1_000_000,
        "tempmem": -1,
        "metric_type": MetricType.inner_product,
        "random_train_subset": False,
    }

    def _build(
        self,
        vectors: Optional[Tensors | TensorArrowTable] = None,
        corpus: Optional[Dataset] = None,
    ):
        """build the index from the vectors."""

        # input checks and casting
        assert len(vectors.shape) == 2, f"The vectors must be 2D. vectors: {vectors.shape}"
        self.config["metric_type"] = MetricType(self.config["metric_type"]).name

        # init the index
        self._index = self._init_index(self.config)

        # train the index
        faiss_train_size = self.config["train_size"]
        random_train_subset = self.config["random_train_subset"]
        if faiss_train_size is not None and faiss_train_size < len(vectors):
            if random_train_subset:
                train_ids = np.random.choice(len(vectors), faiss_train_size, replace=False)
            else:
                train_ids = slice(None, faiss_train_size)
        else:
            train_ids = slice(None, None)

        train_vectors = vectors[train_ids]
        logger.info(
            f"Train the index "
            f"with {len(train_vectors)} vectors "
            f"(type: {type(train_vectors)})."
        )
        self._index.train(train_vectors)

        # add vectors to the index
        logger.info(
            f"Adding {len(vectors)} vectors " f"(type: {type(vectors).__name__}) to the index."
        )
        self._index.add(vectors)

        # free-up GPU memory
        self.cpu()

        # save
        self.save()

    def __len__(self) -> int:
        return self._index.ntotal

    def save(self):
        """save the index to file"""
        super().save()
        self._index.save(self.path)

    def _init_index(self, config: Dict) -> VectorBase:
        # faiss metric
        metric_type = MetricType(config["metric_type"]).name
        faiss_metric = {
            MetricType.inner_product.name: faiss.METRIC_INNER_PRODUCT,
            MetricType.euclidean.name: faiss.METRIC_L2,
        }[metric_type]

        # init the index
        return AutoVectorBase(
            index_factory=config["index_factory"],
            dimension=config["dimension"],
            faiss_metric=faiss_metric,
            nprobe=config["nprobe"],
            train_on_cpu=config["train_on_cpu"],
            keep_on_cpu=config["keep_on_cpu"],
            tempmem=config["tempmem"],
        )

    def load(self):
        """save the index to file"""
        super().load()
        self._index = self._init_index(self.config)
        self._index.load(self.path)

    def cpu(self):
        """Move the index to CPU."""
        self._index.cpu()

    def cuda(self, devices: Optional[List[int]] = None):
        """Move the index to CUDA."""
        self._index.cuda(devices)

    def free_memory(self):
        """Free the memory of the index."""
        self._index = None

    @property
    def is_up(self) -> bool:
        return self._index is not None

    def search(self, *query: Any, k: int = None, **kwargs) -> SearchResult:
        q_vectors, *_ = query
        return self._index.search(q_vectors, k=k)

    def _search_chunk(
        self, query: Batch, *, k: int, vectors: Optional[torch.Tensor], **kwargs
    ) -> SearchResult:
        pprint_batch(query, f"Search:chunk:vectors: {vectors.shape}, dtype: {vectors.dtype}")
        scores, indices = self.search(vectors, k=k)
        return SearchResult(score=scores, index=indices, k=k)
