from os import PathLike
from pathlib import Path
from typing import List
from typing import Optional

import faiss.contrib.torch_utils  # type: ignore
import torch
from loguru import logger

from .base import VectorBase


class ShardedFaissVectorBase(VectorBase):
    def __init__(self, *args, faiss_metric: int = faiss.METRIC_INNER_PRODUCT, **kwargs):
        super().__init__(*args, **kwargs)

        faiss.index_factory(self.dimension, self.index_factory, faiss_metric)

    def train(self, vectors, **kwargs):
        ...

    def add(self, vectors: torch.Tensor, **kwargs):
        ...

    def save(self, path: PathLike):
        ...

    def load(self, path: PathLike):
        ...

    def search(self, query: torch.Tensor, k: int, **kwargs) -> (torch.Tensor, torch.Tensor):
        ...

    @property
    def ntotal(self) -> int:
        ...

    def cuda(self) -> "VectorBase":
        ...

    def cpu(self) -> "VectorBase":
        ...
