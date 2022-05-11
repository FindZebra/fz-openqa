from __future__ import annotations

from typing import List
from typing import Optional
from typing import Tuple

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import torch  # type: ignore
from faiss import IndexReplicas
from loguru import logger

from fz_openqa.datamodules.index.handlers.base import IndexHandler
from fz_openqa.datamodules.index.handlers.vector_base import VectorBase
from fz_openqa.datamodules.index.handlers.vector_base.auto import AutoVectorBase
from fz_openqa.utils.metric_type import MetricType
from fz_openqa.utils.tensor_arrow import TensorArrowTable


class FaissHandler(IndexHandler):
    """This class implements a low level index."""

    def _build(
        self,
        vectors: torch.Tensor | TensorArrowTable | np.ndarray,
        *,
        index_factory: str = "Flat",
        nprobe: int = 32,
        keep_on_cpu: bool = False,
        train_on_cpu: bool = False,
        faiss_train_size: int = None,
        shard_faiss: bool = False,
        metric_type: MetricType = MetricType.inner_product,
        **kwargs,
    ):
        """build the index from the vectors."""
        metric_type = MetricType(metric_type).name
        if isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors)

        assert len(vectors.shape) == 2, f"The vectors must be 2D. vectors: {vectors.shape}"
        self.config["dimension"] = vectors.shape[-1]

        u = (
            f"Setup {type(self).__name__} with "
            f"vectors ({type(vectors)}): ({len(vectors)}, {vectors.shape[-1]})"
        )
        for k, v in [
            ("index_factory", index_factory),
            ("nprobe", nprobe),
            ("keep_on_cpu", keep_on_cpu),
            ("train_on_cpu", train_on_cpu),
            ("faiss_train_size", faiss_train_size),
            ("metric_type", metric_type),
        ]:
            self.config[k] = v
            u += f", {k}={v}"

        # log config
        logger.info(u)

        # init the index
        self._index = self._init_index(self.config)

        # train the index
        if faiss_train_size is not None and faiss_train_size < len(vectors):
            train_ids = np.random.choice(len(vectors), faiss_train_size, replace=False)
        else:
            train_ids = slice(None, None)

        train_vectors = vectors[train_ids]
        logger.info(f"Train the index " f"with {len(train_vectors)} vectors.")
        self._index.train(train_vectors)

        # add vectors to the index
        self._index.add(vectors)

        # free-up GPU memory
        self.cpu()

        # save
        self.save()

    def __len__(self) -> int:
        return self._index.ntotal

    def save(self):
        """save the index to file"""
        super().save()
        self._index.save(self.path)

    def _init_index(self, config) -> VectorBase:

        # faiss metric
        faiss_metric = {
            MetricType.inner_product.name: faiss.METRIC_INNER_PRODUCT,
            MetricType.euclidean.name: faiss.METRIC_L2,
        }[config["metric_type"]]

        # init the index
        return AutoVectorBase(
            index_factory=config["index_factory"],
            dimension=config["dimension"],
            faiss_metric=faiss_metric,
            nprobe=config["nprobe"],
            train_on_cpu=config["train_on_cpu"],
            keep_on_cpu=config["keep_on_cpu"],
        )

    def load(self):
        """save the index to file"""
        super().load()
        self._index = self._init_index(self.config)
        self._index.load(self.path)

    def cpu(self):
        """Move the index to CPU."""
        self._index.cpu()

    def cuda(self, devices: Optional[List[int]] = None):
        """Move the index to CUDA."""
        self._index.cuda(devices)

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
        scores, indexes = self._index.search(query, k)
        return scores, indexes
