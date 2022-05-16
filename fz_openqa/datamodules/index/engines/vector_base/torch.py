from os import PathLike
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import rich
import torch
from torch import nn

from .base import VectorBase
from .utils.faiss import Tensors
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.tensor_arrow import TensorArrowTable


class TorchIndex(nn.Module):
    """Register vectors as parameters,
    return the argmax of the dot product in the forward pass."""

    def __init__(self, dimension: int, **kwargs):
        super().__init__()
        self.register_buffer(
            "device_info",
            torch.zeros(
                1,
            ),
        )
        self.vectors = nn.Parameter(torch.empty(0, dimension), requires_grad=False)

    def add(self, vectors: Tensors, **kwargs):
        device = self.device_info.device
        if isinstance(vectors, TensorArrowTable):
            vectors = vectors[:]
        elif isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors)
        elif isinstance(vectors, torch.Tensor):
            pass
        else:
            raise TypeError(f"unsupported type: {type(vectors)}")
        self.vectors.data = torch.cat([self.vectors.data, vectors.to(device)], dim=0)

    def save(self, path: Path):
        torch.save(self.vectors, path.as_posix())

    def load(self, path: Path):
        vectors = torch.load(path.as_posix())
        self.add(vectors)

    def forward(self, batch: Batch) -> (torch.Tensor, torch.Tensor):
        query: torch.Tensor = batch["query"]
        k: int = batch["k"]
        scores = torch.einsum("bh,nh->bn", query, self.vectors)
        k = min(k, len(self.vectors))
        scores, indices = torch.topk(scores, k, dim=1, largest=True)
        return scores, indices


class TorchVectorBase(VectorBase):
    """A base class for vector index using torch tensors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = TorchIndex(self.dimension)

    def train(self, vectors, **kwargs):
        ...

    def add(self, vectors: torch.Tensor, **kwargs):
        self.base_index.add(vectors)

    @property
    def base_index(self) -> TorchIndex:
        if isinstance(self._model, nn.DataParallel):
            return self._model.module  # type: ignore
        return self._model

    @staticmethod
    def index_file(path: PathLike) -> Path:
        path = Path(path)
        return path / "index.pt"

    def save(self, path: PathLike):
        path = self.index_file(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.base_index.save(path)

    def load(self, path: PathLike):
        path = self.index_file(path)
        if isinstance(self._model, nn.DataParallel):
            devices = self._model.device_ids
            self.base_index.load(path)
            self.cuda(devices)
        else:
            self.base_index.load(path)

    def search(self, query: torch.Tensor, k: int, **kwargs) -> (torch.Tensor, torch.Tensor):
        return self._model({"query": query, "k": k})

    @property
    def ntotal(self) -> int:
        return len(self.base_index.vectors)

    def cuda(self, devices: Optional[List[int]] = None):
        if devices is None:
            devices = list(range(torch.cuda.device_count()))

        if len(devices) == 0:
            return

        self._model = nn.DataParallel(self.base_index, device_ids=devices)
        self._model.cuda()

    def cpu(self):
        self._model = self.base_index
