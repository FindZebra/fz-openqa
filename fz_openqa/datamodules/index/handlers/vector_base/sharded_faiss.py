from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import List
from typing import Optional

import faiss.contrib.torch_utils  # type: ignore
import rich
import torch
from loguru import logger

from .base import VectorBase
from .faiss import FaissVectorBase
from .utils.faiss import build_and_train_ivf_index
from .utils.faiss import compute_centroids
from .utils.faiss import faiss_sanitize
from .utils.faiss import FaissFactory
from .utils.faiss import get_gpu_resources
from .utils.faiss import get_shareded_gpu_index
from .utils.faiss import IdentityVectorTransform
from .utils.faiss import populate_ivf_index
from .utils.faiss import Tensors
from .utils.faiss import train_preprocessor


class ShardedFaissVectorBase(FaissVectorBase):
    def __init__(
        self,
        *,
        tempmem=-1,
        use_float16: bool = True,
        max_add_per_gpu: int = 1 << 25,
        add_batch_size: int = 65536,
        use_precomputed_tables: bool = False,
        replicas: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # paraeters for the index
        self.factory: FaissFactory = FaissFactory(self.index_factory)
        self.tempmem = tempmem
        self.use_float16 = use_float16
        self.max_add_per_gpu = max_add_per_gpu
        self.use_precomputed_tables = use_precomputed_tables
        self.add_batch_size = add_batch_size
        self.replicas = replicas

        # actual index
        self.index = None  # delete the index, and create a new one in `train`
        self.preprocessor = None

    def train(self, vectors: torch.Tensor, **kwargs):

        # use all available GPUs
        gpu_resources = get_gpu_resources(tempmem=self.tempmem)

        # build the preprocessor
        self.preprocessor = self._build_preprocessor(vectors)

        self.index = self._build_ivf_index(gpu_resources, kwargs, vectors)
        self.index.nprobe = self.nprobe

    def add(self, vectors: torch.Tensor, **kwargs):
        # use all available GPUs
        gpu_resources = get_gpu_resources(tempmem=self.tempmem)

        self.index = populate_ivf_index(
            self.index,
            preproc=self.preprocessor,
            vectors=vectors,
            gpu_resources=gpu_resources,
            max_add_per_gpu=self.max_add_per_gpu,
            use_float16=self.use_float16,
            use_precomputed_tables=self.use_precomputed_tables,
            add_batch_size=self.add_batch_size,
        )

    def _build_ivf_index(self, gpu_resources, kwargs, vectors):
        # find the centroids and return a FlatIndex
        coarse_quantizer = self._build_coarse_quantizer(
            vectors, gpu_resources=gpu_resources, **kwargs
        )
        # build the index
        index = build_and_train_ivf_index(
            vectors,
            faiss_factory=self.factory,
            preproc=self.preprocessor,
            coarse_quantizer=coarse_quantizer,
            faiss_metric=self.faiss_metric,
            use_float16=self.use_float16,
        )
        return index

    def _build_preprocessor(
        self, vectors: Tensors
    ) -> faiss.VectorTransform | IdentityVectorTransform:
        if self.factory.preproc is not None:
            return train_preprocessor(self.factory.preproc, vectors=vectors)
        else:
            return IdentityVectorTransform(self.dimension)

    def _build_coarse_quantizer(self, vectors: Tensors, **kwargs) -> faiss.IndexFlat:
        centroids = compute_centroids(
            vectors=vectors,
            preproc=self.preprocessor,
            faiss_metric=self.faiss_metric,
            n_centroids=self.factory.n_centroids,
            **kwargs,
        )

        # build a FlatIndex containing the centroids
        coarse_quantizer = faiss.IndexFlat(self.preprocessor.d_out, self.faiss_metric)
        coarse_quantizer.add(centroids)

        return coarse_quantizer

    @staticmethod
    def index_file(path: PathLike) -> Path:
        path = Path(path)
        return path / "index.faiss"

    @staticmethod
    def preproc_file(path: PathLike) -> Path:
        path = Path(path)
        return path / "preproc.faiss"

    def save(self, path: PathLike):
        index_path = self.index_file(path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path.as_posix())
        preproc_path = self.preproc_file(path)
        if isinstance(self.preprocessor, faiss.VectorTransform):
            faiss.write_VectorTransform(self.preprocessor, preproc_path.as_posix())

    def load(self, path: PathLike):
        index_path = self.index_file(path)
        self.index = faiss.read_index(index_path.as_posix())
        self.index.nprobe = self.nprobe
        preproc_path = self.preproc_file(path)
        if preproc_path.exists():
            self.preprocessor = faiss.read_VectorTransform(preproc_path.as_posix())
        else:
            self.preprocessor = IdentityVectorTransform(self.dimension)

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

        if isinstance(self.index, faiss.IndexShards):
            # the index is already on the correct device
            return

        # copy the cpu index to the GPU shards
        cpu_index = self.index
        self.index = get_shareded_gpu_index(
            cpu_index,
            devices=devices,
            use_float16=self.use_float16,
            use_precomputed_tables=self.use_precomputed_tables,
            replicas=self.replicas,
            tempmem=self.tempmem,
        )
        # make sure to free up the cpu index
        del cpu_index

        # set the nprobe parameter
        ps = faiss.GpuParameterSpace()
        ps.initialize(self.index)
        rich.print(f"[magenta]>> Setting nprobe to {self.nprobe}")
        ps.set_index_parameter(self.index, "nprobe", self.nprobe)

    def cpu(self):
        self.index = faiss.index_gpu_to_cpu(self.index)  # type: ignore

        # set the nprobe parameter
        try:
            self.index.nprobe = self.nprobe
        except Exception as e:
            logger.warning(f"Couldn't set the `nprobe` parameter: {e}")
            pass
