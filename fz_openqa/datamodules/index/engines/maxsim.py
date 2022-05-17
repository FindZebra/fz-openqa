from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import torch  # type: ignore
from datasets import Dataset
from loguru import logger

from .maxsim_utils.maxsim import MaxSim
from fz_openqa.datamodules.index.engines.base import IndexEngine
from fz_openqa.datamodules.index.engines.vector_base import VectorBase
from fz_openqa.datamodules.index.engines.vector_base.auto import AutoVectorBase
from fz_openqa.datamodules.index.engines.vector_base.utils.faiss import TensorLike
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.metric_type import MetricType
from fz_openqa.utils.tensor_arrow import TensorArrowTable


class MaxSimEngine(IndexEngine):
    _max_num_proc: int = 1

    index_columns: List[str] = []
    query_columns: List[str] = []
    no_fingerprint: List[str] = IndexEngine.no_fingerprint + [
        "max_chunksize",
        "max_queue_size",
        "ranker",
        "vectors",
    ]
    no_index_name = IndexEngine.no_index_name + [
        "max_chunksize",
        "max_queue_size",
    ]

    _default_config: Dict[str, Any] = {
        "max_chunksize": 1_000,
        "max_queue_size": 5,
        "metric_type": MetricType.inner_product,
    }
    require_vectors: bool = True

    def _build(
        self,
        vectors: Optional[TensorLike | TensorArrowTable] = None,
        corpus: Optional[Dataset] = None,
    ):
        # check and cast the inputs
        self.config["metric_type"] = MetricType(self.config["metric_type"]).name
        if not isinstance(vectors, TensorArrowTable):
            raise NotImplementedError("MaxSimEngine requires TensorArrowTable")

        self.vectors = vectors
        self.config["vectors_path"] = self.vectors.path.as_posix()

    def load(self):
        super(MaxSimEngine, self).load()
        self.ranker = self._init_index(self.config)

    def _init_index(self, config, devices=None):
        if self.vectors is None or not hasattr(self, "vectors"):
            self.vectors = TensorArrowTable(path=config["vectors_path"])

        return MaxSim(
            vectors=self.vectors,
            devices=devices,
            max_chunksize=config["max_chunksize"],
            max_queue_size=config["max_queue_size"],
            metric_type=config["metric_type"],
        )

    def cpu(self):
        self.ranker.cpu()

    def cuda(self, devices: Optional[List[int]] = None):
        if devices is not None and devices != self.ranker.devices:
            del self.ranker
            self.ranker = self._init_index(self.config, devices=devices)
        else:
            self.ranker.cuda(device=devices)

    @property
    def is_up(self) -> bool:
        return hasattr(self, "ranker") and self.ranker is not None

    def free_memory(self):
        if hasattr(self, "ranker"):
            del self.ranker

    def search(self, *query: Any, k: int = None, **kwargs) -> (TensorLike, TensorLike):
        q_vectors, pids, *_ = query
        output = self.ranker(q_vectors, k=k, pids=pids)
        return output.scores, output.pids

    def _search_chunk(
        self,
        query: Batch,
        *,
        k: int,
        vectors: Optional[torch.Tensor],
        pids: Optional[TensorLike] = None,
        **kwargs,
    ) -> SearchResult:

        if pids is None:
            raise ValueError("pids is required")
        elif isinstance(pids, list):
            pids = torch.tensor(pids)
        elif isinstance(pids, np.ndarray):
            pids = torch.from_numpy(pids)
        elif isinstance(pids, torch.Tensor):
            pass
        else:
            raise TypeError("pids must be a list, numpy array or torch tensor")

        scores, indices = self.search(vectors, pids, k=k)
        return SearchResult(score=scores, index=indices, k=k)
