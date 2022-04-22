from __future__ import annotations

import abc
from collections import defaultdict
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import torch
from loguru import logger

from fz_openqa.datamodules.index.handlers.base import IndexHandler
from fz_openqa.utils.tensor_arrow import TensorArrowTable


class IndexLookupHandler(IndexHandler):
    """Retrieve all the passages corresponding to a given document id."""

    def _build(
        self,
        vectors: torch.Tensor | TensorArrowTable | np.ndarray,
        *,
        doc_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        """build the index from the vectors."""
        if doc_ids is None:
            raise ValueError(f"`doc_ids` is required to build the" f"{self.__class__.__name__}")

        if not len(doc_ids) == len(vectors):
            raise ValueError(
                f"`doc_ids` and `vectors` must have the same length. "
                f"Found {len(doc_ids)} and {len(vectors)} respectively."
            )

        # NB: for Colbert, this is a pretty ineffective way to build the index: this is an index
        # from document id to token id, an index from document id to passage id would be better.
        lookup_ = defaultdict(list)
        for tokid, doc_id in enumerate(doc_ids):
            lookup_[doc_id].append(tokid)

        n_cols = max(len(v) for v in lookup_.values()) + 1
        n_rows = max(lookup_.keys()) + 1

        # todo: need the handle the -1 on MaxSim side
        self._lookup = torch.empty(n_rows, n_cols, dtype=torch.int64).fill_(-1)
        for i in range(n_rows):
            self._lookup[i, : len(lookup_[i])] = torch.tensor(sorted(lookup_[i]))

        logger.info(f"Lookup table: {self._lookup.shape}")
        self.save()

    def __len__(self) -> int:
        return self._lookup.shape[1]

    @property
    def lookup_file(self) -> Path:
        return self.path / "lookup.pt"

    def save(self):
        """save the index to file"""
        super().save()
        torch.save(self._lookup, self.lookup_file)

    def load(self):
        """save the index to file"""
        super().load()
        self._lookup = torch.load(self.lookup_file)

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
        self._lookup = None

    @property
    def is_up(self) -> bool:
        return self._lookup is not None

    def __del__(self):
        self.free_memory()

    def __call__(
        self, query: torch.Tensor, *, k: int, doc_ids: List[int] | torch.Tensor = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if doc_ids is None:
            raise ValueError("`doc_ids` is required")

        if not len(doc_ids) == query.shape[0]:
            raise ValueError(
                f"`doc_ids` and `vectors` must have the same length. "
                f"Found {len(doc_ids)} and {query.shape[0]} respectively."
            )

        doc_ids = torch.tensor(doc_ids, dtype=torch.int64, device=self._lookup.device)

        pids = self._lookup[doc_ids]
        scores = torch.zeros_like(pids, dtype=torch.float32)
        return scores, pids
