from __future__ import annotations

from dataclasses import dataclass
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


class GpuConfig:
    def __init__(self, ngpu: int):
        self.ngpu = ngpu
        self.tempmem = 1 << 33
        self.max_add_per_gpu = 1 << 25
        self.max_add = self.max_add_per_gpu * ngpu
        self.add_batch_size = 65536

    def __repr__(self):
        return (
            f"GpuConfig(ngpu={self.ngpu}, "
            f"tempmem={self.tempmem}, "
            f"max_add_per_gpu={self.max_add_per_gpu}, "
            f"max_add={self.max_add}, "
            f"add_batch_size={self.add_batch_size})"
        )


class TorchIndex(nn.Module):
    """A placeholder index to replace a faiss index using torch tensors."""

    def train(self, vectors, **kwargs):
        ...

    def add(self, vectors: torch.Tensor, **kwargs):
        if hasattr(self, "vectors"):
            self.vectors = torch.cat([self.vectors, vectors], dim=0)
        else:
            self.register_buffer("vectors", vectors)

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
        faiss_train_size: int = None,
        **kwargs,
    ):
        """build the index from the vectors."""
        if isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors)

        assert len(vectors.shape) == 2, f"The vectors must be 2D. vectors: {vectors.shape}"
        self.dimension = vectors.shape[-1]
        self.index_factory = index_factory
        self.keep_on_cpu = keep_on_cpu
        self.train_on_cpu = train_on_cpu
        logger.info(
            f"Setup {type(self).__name__} with "
            f"keep_on_cpu={self.keep_on_cpu}, "
            f"train_on_cpu={self.train_on_cpu}, "
            f"index_factory={self.index_factory}, "
            f"nprobe={self.config.get('nprobe', None)}, "
            f"vectors ({type(vectors)}): {len(vectors)},"
            f"faiss_train_size={faiss_train_size}"
        )

        # init the index
        if index_factory == "torch":
            self._index = TorchIndex()
        else:
            self._index = faiss.index_factory(
                self.dimension, index_factory, faiss.METRIC_INNER_PRODUCT
            )

        # set `nprobe`
        self._index.nprobe = self.config.get("nprobe", None)

        # move the index to GPU
        if not self.train_on_cpu:
            self.cuda()

        # train the index
        if faiss_train_size is not None and faiss_train_size < len(vectors):
            train_ids = np.random.choice(len(vectors), faiss_train_size, replace=False)
        else:
            train_ids = slice(None, None)

        train_vectors = vectors[train_ids]
        logger.info(f"Train the index " f"with {len(train_vectors)} vectors.")
        train_vectors = train_vectors.to(torch.float32)
        self._index.train(train_vectors)

        # add vectors to the index
        batch_size = faiss_train_size or len(vectors)
        logger.info(f"Adding {len(vectors)} to the index " f"with batch size {batch_size}.")
        for i in range(0, len(vectors), batch_size):
            vecs = vectors[i : i + batch_size]
            vecs = vecs.to(torch.float32)
            self._index.add(vecs)

        # free-up GPU memory
        self.cpu()

        # save
        self.save()

    def __len__(self) -> int:
        return self._index.ntotal

    @property
    def index_file(self) -> Path:
        return self.path / "index.faiss"

    def save(self):
        """save the index to file"""
        super().save()
        if not isinstance(self._index, TorchIndex):
            faiss.write_index(self._index, str(self.index_file))
        else:
            self._index.save(self.index_file)

    def load(self):
        """save the index to file"""
        super().load()
        if self.config.get("index_factory") == "torch":
            self._index = TorchIndex()
            self._index.load(self.index_file)
        else:
            self._index = faiss.read_index(str(self.index_file))

    def cpu(self):
        """Move the index to CPU."""
        try:
            self._index = faiss.index_gpu_to_cpu(self._index)  # type: ignore
            try:
                self._index.nprobe = self.config.get("nprobe", None)
            except Exception as e:
                logger.warning(f"Couldn't set the `nprobe` parameter: {e}")
                pass
        except Exception:
            pass

    def _build_gpu_index(self, index, devices: List[int] = None):
        if devices is None or len(devices) == 0:
            raise ValueError("devices must be a list of GPU ids")

        # GPU configuration
        gpu_resources, gpu_cfg = self._prepare_gpu_resources(devices)
        vres, vdev = self._make_vres_vdev(gpu_resources)
        co = self._make_cloner_options(gpu_cfg)
        return faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

    def cuda(self, devices: Optional[List[int]] = None):
        """Move the index to CUDA."""
        keep_on_cpu = self.config.get("keep_on_cpu", None)
        if keep_on_cpu or isinstance(faiss, IndexReplicas):
            return

        if devices is None:
            devices = list(range(faiss.get_num_gpus()))

        if len(devices) == 0:
            return

        # move the index to GPU
        self._index = self._build_gpu_index(self._index, devices)

        # set `nprobe`
        nprobe = self.config.get("nprobe", None)
        if nprobe is not None:

            # retrieve the coarse quantizer index (IVF, IMI, ...)
            try:
                ivf_index = faiss.extract_index_ivf(self._index)
            except Exception as e:
                logger.warning(e)
                ivf_index = self._index

            # set the nprobe parameter
            try:
                gspace = faiss.GpuParameterSpace()  # type: ignore
                gspace.set_index_parameter(ivf_index, "nprobe", nprobe)
            except Exception as e:
                logger.warning(f"Couldn't set the `nprobe` parameter: {e}")
                try:
                    ivf_index.nprobe = nprobe
                except Exception as e:
                    logger.warning(f"Couldn't set the `nprobe` parameter: {e}")
                    pass
        else:
            logger.warning("Parameter `nprobe` is not set")

        logger.info(
            f"to-cuda end: is_trained: {self._index.is_trained}," f" ntotal={self._index.ntotal}"
        )

    def free_memory(self):
        """Free the memory of the index."""
        self._index = None

    @property
    def is_up(self) -> bool:
        return self._index is not None

    def __del__(self):
        self.free_memory()

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

    def _prepare_gpu_resources(self, devices: List[int]):
        ngpu = len(devices)
        gpu_resources = []

        gpu_config = GpuConfig(ngpu)
        logger.info(f"GPU configuration: {gpu_config}")

        for _ in range(ngpu):
            res = faiss.StandardGpuResources()
            if gpu_config.tempmem >= 0:
                res.setTempMemory(gpu_config.tempmem)
            gpu_resources.append(res)

        return gpu_resources, gpu_config

    def _make_vres_vdev(self, gpu_resources: List):
        """
        return vectors of device ids and resources useful for gpu_multiple
        """
        ngpu = len(gpu_resources)
        assert ngpu > 0

        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()

        for i in range(ngpu):
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])

        return vres, vdev

    def _make_cloner_options(self, gpu_cfg) -> faiss.GpuMultipleClonerOptions:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        co.useFloat16CoarseQuantizer = False
        co.usePrecomputed = False
        co.indicesOptions = faiss.INDICES_CPU
        co.verbose = True
        co.reserveVecs = gpu_cfg.max_add
        co.shard = True
        assert co.shard_type in (0, 1, 2)
        return co
