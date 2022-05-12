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
from .utils.faiss import build_and_train_ivf_index
from .utils.faiss import compute_centroids
from .utils.faiss import faiss_sanitize
from .utils.faiss import FaissFactory
from .utils.faiss import get_gpu_resources
from .utils.faiss import get_sharded_gpu_index
from .utils.faiss import IdentityVectorTransform
from .utils.faiss import populate_ivf_index
from .utils.faiss import Tensors
from .utils.faiss import train_preprocessor


class FaissVectorBase(VectorBase):
    def __init__(
        self,
        *,
        nprobe: int,
        faiss_metric: int = faiss.METRIC_INNER_PRODUCT,
        train_on_cpu: bool = False,
        keep_on_cpu: bool = False,
        tempmem=-1,
        use_float16: bool = True,
        max_add_per_gpu: int = 1 << 25,
        add_batch_size: int = 65536,
        use_precomputed_tables: bool = False,
        replicas: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if self.index_factory.startswith("shard:"):
            self.shard = True
            self.index_factory = self.index_factory.replace("shard:", "")
        else:
            self.shard = False

        # parameters for the index
        self.nprobe = nprobe
        self.faiss_metric = faiss_metric
        self.factory: FaissFactory = FaissFactory(self.index_factory)
        self.tempmem = tempmem
        self.use_float16 = use_float16
        self.max_add_per_gpu = max_add_per_gpu
        self.use_precomputed_tables = use_precomputed_tables
        self.add_batch_size = add_batch_size
        self.replicas = replicas

        logger.info(
            f"Using {type(self).__name__}({self.index_factory}, "
            f"shard={self.shard}, "
            f"nprobe={nprobe})"
        )
        logger.info(self.factory)

        # index attributes
        self.index = None  # delete the index, and create a new one in `train`
        self.preprocessor = None

        # todo: implement functionalities for keeping the index on CPU
        self.train_on_cpu = train_on_cpu
        self.keep_on_cpu = keep_on_cpu

    def train(self, vectors: torch.Tensor, **kwargs):

        # use all available GPUs
        gpu_resources = get_gpu_resources(tempmem=self.tempmem)

        # build the preprocessor
        self.preprocessor = self._build_preprocessor(vectors)

        if self.train_on_cpu or len(gpu_resources) == 0:
            self.index = self._build_ivf_index_cpu(vectors)
        else:
            self.index = self._build_ivf_index(gpu_resources, vectors, **kwargs)

        # set nprobe
        self.index.nprobe = self.nprobe

    def add(self, vectors: torch.Tensor, **kwargs):
        # use all available GPUs
        gpu_resources = get_gpu_resources(tempmem=self.tempmem)

        if self.train_on_cpu or len(gpu_resources) == 0:
            self.index = self._populate_ivf_index_cpu(self.index, vectors=vectors)

        else:
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

        # set nprobe
        self.index.nprobe = self.nprobe

    def _build_ivf_index(self, gpu_resources, vectors, **kwargs):
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

    def _build_ivf_index_cpu(self, vectors):
        index = faiss.index_factory(self.dimension, self.index_factory, self.faiss_metric)
        vectors = faiss_sanitize(vectors)
        index.train(vectors)
        return index

    def _populate_ivf_index_cpu(self, index, *, vectors):
        for i in range(0, len(vectors), self.add_batch_size):
            v = vectors[i : i + self.add_batch_size]
            v = faiss_sanitize(v, force_numpy=True)
            index.add(v)
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
        try:
            query = self.preprocessor.apply_py(query)
        except Exception as exc:
            logger.error(
                f"Failed to preprocess query ({type(query)}) "
                f"with preprocessor: {self.preprocessor}. "
                f"Exception: {exc}"
            )

        logger.debug(f"Faiss:Query: {query.shape}, k: {k}")
        return self.index.search(query, k)

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def _move_to_cuda_shard(self, index: faiss.Index, devices: List[int]) -> faiss.IndexShards:
        return get_sharded_gpu_index(
            index,
            devices=devices,
            use_float16=self.use_float16,
            use_precomputed_tables=self.use_precomputed_tables,
            replicas=self.replicas,
            tempmem=self.tempmem,
        )

    @staticmethod
    def _mode_to_cuda(index: faiss.Index, devices: List[int]) -> faiss.IndexShards:
        return faiss.index_cpu_to_gpus_list(index, gpus=devices)

    def cuda(self, devices: Optional[List[int]] = None):

        if self.keep_on_cpu:
            return

        if devices is None:
            devices = list(range(faiss.get_num_gpus()))

        if len(devices) == 0:
            return

        # move the index to the GPU
        if self.shard:
            logger.warning(f">> Moving index to GPU shard {devices}")
            self.index = self._mode_to_cuda(self.index, devices)
        else:
            logger.warning(f">> Moving index to GPU {devices} (no sharding)")
            self.index = self._move_to_cuda_shard(self.index, devices)

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
