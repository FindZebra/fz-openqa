from os import PathLike
from pathlib import Path
from typing import List
from typing import Optional

import faiss.contrib.torch_utils  # type: ignore
import rich
import torch
from loguru import logger

from .base import VectorBase


class FaissVectorBase(VectorBase):
    def __init__(
        self,
        *args,
        nprobe: int,
        faiss_metric: int = faiss.METRIC_INNER_PRODUCT,
        train_on_cpu: bool = False,
        keep_on_cpu: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nprobe = nprobe
        self.index = faiss.index_factory(self.dimension, self.index_factory, faiss_metric)
        self.index.nprobe = nprobe

        # todo: implement functionalities for keeping the index on CPU
        self.train_on_cpu = train_on_cpu
        self.keep_on_cpu = keep_on_cpu

    def train(self, vectors: torch.Tensor, **kwargs):
        vectors = self._sanitize(vectors)
        return self.index.train(vectors)

    def add(self, vectors: torch.Tensor, **kwargs):
        vectors = self._sanitize(vectors)
        self.index.add(vectors)

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

    def search(self, query: torch.Tensor, k: int, **kwargs) -> (torch.Tensor, torch.Tensor):
        query = self._sanitize(query)
        return self.index.search(query, k)

    @staticmethod
    def _sanitize(x: torch.Tensor) -> torch.Tensor:
        rich.print(f"> x: {type(x)}, {x.shape}")
        x = x.to(torch.float32)
        return x

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def cuda(self, devices: Optional[List[int]] = None):
        if devices is None:
            devices = list(range(faiss.get_num_gpus()))

        if len(devices) == 0:
            return

        self.index = faiss.index_cpu_to_gpus_list(self.index, gpus=devices)

        # retrieve the coarse quantizer index (IVF, IMI, ...)
        try:
            ivf_index = faiss.extract_index_ivf(self.index)
        except Exception as e:
            logger.warning(e)
            ivf_index = self.index

        # set the nprobe parameter
        try:
            gspace = faiss.GpuParameterSpace()  # type: ignore
            gspace.set_index_parameter(ivf_index, "nprobe", self.nprobe)
        except Exception as e:
            logger.warning(f"Couldn't set the `nprobe` parameter: {e}")
            try:
                ivf_index.nprobe = self.nprobe
            except Exception as e:
                logger.warning(f"Couldn't set the `nprobe` parameter: {e}")
                pass

    def cpu(self):
        self.index = faiss.index_gpu_to_cpu(self.index)  # type: ignore

        # set the nprobe parameter
        try:
            self.index.nprobe = self.nprobe
        except Exception as e:
            logger.warning(f"Couldn't set the `nprobe` parameter: {e}")
            pass
