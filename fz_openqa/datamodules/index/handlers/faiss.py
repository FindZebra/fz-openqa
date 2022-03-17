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
from loguru import logger
from torch import nn

from fz_openqa.datamodules.index.handlers.base import IndexHandler
from fz_openqa.utils.tensor_arrow import TensorArrowTable


class TorchIndex(nn.Module):
    """A placeholder index to replace a faiss index using torch tensors."""

    def train(self, vectors, **kwargs):
        ...

    def add(self, vectors: torch.Tensor, **kwargs):
        self.register_buffer("vectors", vectors.to(torch.float32))

    def save(self, path: Path):
        torch.save(self.vectors, path)

    def load(self, path: Path):
        vectors = torch.load(path)
        self.add(vectors)

    def search(self, query: torch.Tensor, k: int, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.einsum("bh,nh->bn", query, self.vectors)
        k = min(k, self.ntotal)
        scores, indices = torch.topk(scores, k, dim=1, largest=True)
        return scores, indices

    @property
    def ntotal(self) -> int:
        return len(self.vectors)


class FaissHandler(IndexHandler):
    """This class implements a low level index."""

    def _build(
        self,
        vectors: torch.Tensor | TensorArrowTable | np.ndarray,
        *,
        index_factory: str = "Flat",
        nprobe: int = 8,
        keep_on_cpu: bool = False,
        train_on_cpu: bool = False,
        **kwargs,
    ):
        """build the index from the vectors."""
        if isinstance(vectors, TensorArrowTable):
            vectors = vectors[:]
        elif isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors)

        assert vectors.dim() == 2, f"The vectors must be 2D. vectors: {vectors.shape}"
        self.dimension = vectors.shape[-1]
        self.index_factory = index_factory
        self.keep_on_cpu = keep_on_cpu
        self.train_on_cpu = train_on_cpu
        logger.info(
            f"Setup {type(self).__name__} with "
            f"keep_on_cpu={self.keep_on_cpu}, "
            f"train_on_cpu={self.train_on_cpu}, "
            f"index_factory={self.index_factory}, "
            f"nprobe={nprobe}, "
            f"vectors: {vectors.shape}"
        )

        # init the index
        if index_factory == "torch":
            self._index = TorchIndex()
        else:
            self._index = faiss.index_factory(
                self.dimension, index_factory, faiss.METRIC_INNER_PRODUCT
            )

        # set `nprobe`
        self._index.nprobe = nprobe

        # move the index to GPU
        if not self.train_on_cpu:
            self.cuda()

        # add vectors to the index
        # todo: avoid casting to float32
        if not isinstance(self._index, TorchIndex):
            vectors = vectors.to(torch.float32)
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
        if not isinstance(self._index, TorchIndex):
            faiss.write_index(self._index, str(self.index_file))
        else:
            self._index.save(self.index_file)

    @abc.abstractmethod
    def load(self):
        """save the index to file"""
        super().load()
        if self.config.get("index_factory") == "torch":
            self._index = TorchIndex()
            self._index.load(self.index_file)
        else:
            self._index = faiss.read_index(str(self.index_file))
        self.keep_on_cpu = self.config.get("keep_on_cpu", None)

    @abc.abstractmethod
    def cpu(self):
        """Move the index to CPU."""
        try:
            self._index = faiss.index_gpu_to_cpu(self._index)  # type: ignore
        except Exception:
            pass

    @abc.abstractmethod
    def cuda(self, devices: Optional[List[int]] = None):
        """Move the index to CUDA."""
        if self.keep_on_cpu or isinstance(faiss, IndexReplicas):
            return

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
        self._index = faiss.index_cpu_to_gpus_list(self._index, gpus=devices)

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
        if not isinstance(self._index, TorchIndex):
            query = query.to(torch.float32)
        else:
            query = query.to(self._index.vectors)
        scores, indexes = self._index.search(query, k)
        return scores, indexes