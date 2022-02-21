from __future__ import annotations

import abc
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import torch
from faiss import IndexReplicas

from fz_openqa.datamodules.index.handlers.base import IndexHandler
from fz_openqa.utils.tensor_arrow import TensorArrowTable


class FaissHandler(IndexHandler):
    """This class implements a low level index."""

    def _build(
        self,
        vectors: torch.Tensor | TensorArrowTable | np.ndarray,
        *,
        index_factory: str = "Flat",
        nprobe: int = 8,
        **kwargs,
    ):
        """build the index from the vectors."""
        if isinstance(vectors, TensorArrowTable):
            vectors = vectors[:]
        elif isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors)

        assert vectors.dim() == 2, f"The vectors must be 2D. vectors: {type(vectors)}"
        self.dimension = vectors.shape[-1]
        self.index_factory = index_factory

        # init the index
        self._index = faiss.index_factory(self.dimension, index_factory, faiss.METRIC_INNER_PRODUCT)

        # set `nprobe`
        self._index.nprobe = nprobe

        # move the index to GPU
        self.cuda()

        # add vectors to the index
        self._index.train(vectors)
        self._index.add(vectors)

        # free-up GPU memory
        self.cpu()

        # save
        self.save()

    def __len__(self) -> int:
        return self._index.ntotal

    @property
    def index_file(self) -> Path:
        return self.path / "index.faiss"

    @abc.abstractmethod
    def save(self):
        """save the index to file"""
        super().save()
        faiss.write_index(self._index, str(self.index_file))

    @abc.abstractmethod
    def load(self):
        """save the index to file"""
        super().load()
        self._index = faiss.read_index(str(self.index_file))

    @abc.abstractmethod
    def cpu(self):
        """Move the index to CPU."""
        if isinstance(faiss, IndexReplicas):
            self._index = faiss.index_gpu_to_cpu(self._index)  # type: ignore

    @abc.abstractmethod
    def cuda(self, devices: Optional[List[int]] = None):
        """Move the index to CUDA."""
        if devices is None:
            devices = list(range(faiss.get_num_gpus()))

        if len(devices) == 0:
            return

        # register `nprobe`
        try:
            nprobe = self._index.nprobe
        except Exception:
            nprobe = None

        # move the index to GPU
        self._index = faiss.index_cpu_to_gpus_list(self._index, devices)

        # set `nprobe`
        if nprobe is not None:
            gspace = faiss.GpuParameterSpace()  # type: ignore
            gspace.set_index_parameter(self._index, "nprobe", nprobe)

    @abc.abstractmethod
    def free_memory(self):
        """Free the memory of the index."""
        self._index = None

    @property
    def is_up(self) -> bool:
        return self._index is not None

    def __del__(self):
        self.free_memory()

    @abc.abstractmethod
    def __call__(
        self, query: torch.Tensor, *, k: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call the index."""
        return self._index.search(query, k)
