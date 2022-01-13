from __future__ import annotations

from typing import List

import faiss
import numpy as np
import torch
from faiss.swigfaiss import Index as FaissSwigIndex

from fz_openqa.datamodules.index.utils.maxsim.datastruct import FaissInput
from fz_openqa.utils.datastruct import PathLike


class TokenIndex(object):
    """This class handle a token-level faiss index"""

    def __init__(
        self,
        index: faiss.Index | PathLike,
        devices: List[int],
        multiprocessing: bool = False,
        **kwargs,
    ):

        self.devices = [i for i in set(devices) if i >= 0]
        super(TokenIndex, self).__init__()
        assert isinstance(devices, list)
        if multiprocessing and isinstance(index, faiss.Index):
            self._index = faiss.serialize_index(index)
        elif isinstance(index, faiss.Index):
            self._index = index
        else:
            self._index = str(index)

        if not multiprocessing:
            self.prepare()

    def prepare(self):
        if isinstance(self._index, str):
            self._index = faiss.read_index(self._index)
        elif isinstance(self._index, np.ndarray):
            self._index = faiss.deserialize_index(self._index)

    @property
    def index(self) -> FaissSwigIndex:
        if not isinstance(self._index, FaissSwigIndex):
            self.prepare()
        return self._index

    def cuda(self):
        if len(self.devices) > 0 and not isinstance(self.index, faiss.IndexReplicas):
            try:
                nprobe = self.index.nprobe
            except Exception:
                nprobe = None

            self._index = faiss.index_cpu_to_gpus_list(self.index, gpus=self.devices)
            if nprobe is not None:
                params = faiss.GpuParameterSpace()  # type: ignore
                params.set_index_parameter(self._index, "nprobe", nprobe)

    def cpu(self):
        try:
            self._index = faiss.index_gpu_to_cpu(self.index)  # type: ignore
        except Exception:
            pass

    def process_data(self, data):
        # rich.print(f"> {type(self).__name__}(id={self.id})")
        assert isinstance(data, FaissInput)
        token_ids = TokenIndex._query_to_embedding_ids(data.q_vectors, data.p, index=self.index)
        return token_ids

    def __call__(self, q_vectors: torch.Tensor, p: int) -> torch.Tensor:
        return self.process_data(FaissInput(q_vectors, p))

    def cleanup(self):
        del self._index

    @staticmethod
    @torch.no_grad()
    def _query_to_embedding_ids(
        Q: torch.Tensor, faiss_depth: int, *, index: faiss.Index
    ) -> torch.Tensor:
        """Query the faiss index for each embedding vector"""
        num_queries, embeddings_per_query, dim = Q.shape
        Q = Q.view(-1, dim)
        _, embedding_ids = index.search(Q.to(torch.float32), faiss_depth)
        if isinstance(embedding_ids, np.ndarray):
            embedding_ids = torch.from_numpy(embedding_ids)
        embedding_ids = embedding_ids.view(num_queries, -1)
        return embedding_ids.to(Q.device)
