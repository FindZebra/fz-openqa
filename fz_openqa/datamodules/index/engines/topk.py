from __future__ import annotations

from typing import Any
from typing import List
from typing import Optional

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import torch  # type: ignore
from datasets import Dataset
from loguru import logger

from fz_openqa.datamodules.index.engines.base import IndexEngine
from fz_openqa.datamodules.index.engines.vector_base.utils.faiss import TensorLike
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.tensor_arrow import TensorArrowTable


class TopkEngine(IndexEngine):
    _max_num_proc: int = None
    require_vectors: bool = False
    no_fingerprint: List[str] = IndexEngine.no_fingerprint + [
        "merge_previous_results",
    ]

    def _build(
        self,
        vectors: Optional[TensorLike | TensorArrowTable] = None,
        corpus: Optional[Dataset] = None,
    ):
        # check and cast the inputs
        if self.merge_previous_results:
            logger.warning(
                "merge_previous_results is set to True, "
                "but MaxSimEngine is a re-ranker: setting "
                "merge_previous_results to False"
            )
            self.merge_previous_results = False

    def cpu(self):
        pass

    def cuda(self, devices: Optional[List[int]] = None):
        pass

    @property
    def is_up(self) -> bool:
        return True

    def free_memory(self):
        pass

    def search(self, *query: Any, k: int = None, **kwargs) -> (TensorLike, TensorLike):
        scores, pids, *_ = query
        idx = torch.argsort(scores, descending=True, dim=1)[:, :k]
        scores = scores.gather(1, index=idx)
        pids = pids.gather(1, index=idx)
        return scores, pids

    def _search_chunk(
        self,
        query: Batch,
        *,
        k: int,
        vectors: Optional[torch.Tensor],
        scores: Optional[torch.Tensor] = None,
        pids: Optional[TensorLike] = None,
        **kwargs,
    ) -> SearchResult:

        pids = self._check_input(pids, "pids")
        scores = self._check_input(scores, "scores")

        scores, indices = self.search(scores, pids, k=k)

        # here, fill_missing_values=True so -1 indices are filled with random indices
        return SearchResult(score=scores, index=indices, k=k, fill_missing_values=True)

    def _check_input(self, pids, key):
        if pids is None:
            raise ValueError(f"{key} is required")
        elif isinstance(pids, list):
            pids = torch.tensor(pids)
        elif isinstance(pids, np.ndarray):
            pids = torch.from_numpy(pids)
        elif isinstance(pids, torch.Tensor):
            pass
        else:
            raise TypeError(f"{key} must be a list, numpy array or torch tensor")
        return pids
