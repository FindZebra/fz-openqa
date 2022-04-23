from __future__ import annotations

import abc
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import rich
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
        train_vectors = train_vectors.to(torch.float32)
        self._index.train(train_vectors)

        # add vectors to the index
        batch_size = faiss_train_size or len(vectors)
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
        self._index = faiss.index_cpu_to_gpus_list(self._index, gpus=devices)

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
