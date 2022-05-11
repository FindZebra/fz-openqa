from os import PathLike
from pathlib import Path
from typing import List
from typing import Optional

import faiss.contrib.torch_utils  # type: ignore
import rich
import torch
from loguru import logger

from .base import VectorBase
from .utils.faiss import faiss_sanitize


class FaissVectorBase(VectorBase):
    def __init__(
        self,
        *,
        nprobe: int,
        faiss_metric: int = faiss.METRIC_INNER_PRODUCT,
        train_on_cpu: bool = False,
        keep_on_cpu: bool = False,
        add_batch_size: int = 65536,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nprobe = nprobe
        self.faiss_metric = faiss_metric
        self.index = faiss.index_factory(self.dimension, self.index_factory, faiss_metric)
        self.index.nprobe = nprobe
        self.add_batch_size = add_batch_size

        # todo: implement functionalities for keeping the index on CPU
        self.train_on_cpu = train_on_cpu
        self.keep_on_cpu = keep_on_cpu

    def train(self, vectors: torch.Tensor, **kwargs):
        vectors = faiss_sanitize(vectors)
        return self.index.train(vectors)

    def add(self, vectors: torch.Tensor, **kwargs):
        for i in range(0, len(vectors), self.add_batch_size):
            v = vectors[i : i + self.add_batch_size]
            v = faiss_sanitize(v)
            self.index.add(v)

    @staticmethod
    def index_file(path: PathLike) -> Path:
        path = Path(path)
        return path / "index.faiss"

    def save(self, path: PathLike):
        path = self.index_file(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path.as_posix())

    def load(self, path: PathLike):
        path = self.index_file(path)
        self.index = faiss.read_index(path.as_posix())
        self.index.nprobe = self.nprobe

    def search(self, query: torch.Tensor, k: int, **kwargs) -> (torch.Tensor, torch.Tensor):
        query = faiss_sanitize(query)
        return self.index.search(query, k)

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def cuda(self, devices: Optional[List[int]] = None):
        if devices is None:
            devices = list(range(faiss.get_num_gpus()))

        if len(devices) == 0:
            return

        # move the index to the GPU
        self.index = faiss.index_cpu_to_gpus_list(self.index, gpus=devices)

        # set the nprobe parameter
        ps = faiss.GpuParameterSpace()
        ps.initialize(self.index)
        ps.set_index_parameter(self.index, "nprobe", self.nprobe)

    def cpu(self):
        self.index = faiss.index_gpu_to_cpu(self.index)  # type: ignore

        # set the nprobe parameter
        try:
            self.index.nprobe = self.nprobe
        except Exception as e:
            logger.warning(f"Couldn't set the `nprobe` parameter: {e}")
            pass
