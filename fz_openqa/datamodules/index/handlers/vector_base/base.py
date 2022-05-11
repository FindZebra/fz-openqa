import abc
from os import PathLike
from pathlib import Path
from typing import List
from typing import Optional

import torch


class VectorBase:
    """A class to handle the indexing of vectors."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, *, index_factory: str, dimension: int, **kwargs):
        self.index_factory = index_factory
        self.dimension = dimension

    @abc.abstractmethod
    def train(self, vectors, **kwargs):
        ...

    @abc.abstractmethod
    def add(self, vectors: torch.Tensor, **kwargs):
        ...

    @abc.abstractmethod
    def save(self, path: PathLike):
        ...

    @abc.abstractmethod
    def load(self, path: PathLike):
        ...

    @abc.abstractmethod
    def search(self, query: torch.Tensor, k: int, **kwargs) -> (torch.Tensor, torch.Tensor):
        ...

    @property
    def ntotal(self) -> int:
        ...

    @abc.abstractmethod
    def cuda(self, devices: Optional[List[int]] = None):
        ...

    @abc.abstractmethod
    def cpu(self):
        ...
