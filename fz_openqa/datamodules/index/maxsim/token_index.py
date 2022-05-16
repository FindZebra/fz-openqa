from __future__ import annotations

from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import torch

from fz_openqa.datamodules.index.engines.base import IndexEngine
from fz_openqa.datamodules.index.maxsim.datastruct import FaissInput
from fz_openqa.utils.datastruct import PathLike


class TokenIndex(object):
    """This class handle a token-level faiss index"""

    def __init__(
        self,
        index: IndexEngine | PathLike,
        devices: List[int],
        multiprocessing: bool = False,
        max_chunksize: int = None,
        **kwargs,
    ):
        self.devices = [i for i in set(devices) if i >= 0]
        super(TokenIndex, self).__init__()
        assert isinstance(devices, list)

        if not isinstance(index, IndexEngine):
            assert isinstance(index, (str, Path))
        self._index: IndexEngine | PathLike = index
        self.max_chunksize = max_chunksize

        if not multiprocessing:
            self.prepare()

    def prepare(self):
        if isinstance(self._index, (str, Path)):
            self._index = IndexEngine.load_from_path(self._index)
            self._index.load()

    @property
    def index(self) -> IndexEngine:
        if not isinstance(self._index, IndexEngine):
            self.prepare()
        return self._index

    def cuda(self):
        self.index.cuda(devices=self.devices)

    def cpu(self):
        self.index.cpu()

    def process_data(self, data):
        # rich.print(f"> {type(self).__name__}(id={self.id})")
        assert isinstance(data, FaissInput)
        q_vectors = data.q_vectors
        if self.max_chunksize is None:
            max_batch_size = len(q_vectors)
        else:
            max_batch_size = max(1, self.max_chunksize // q_vectors.shape[1])
        token_ids = None
        for i in range(0, q_vectors.size(0), max_batch_size):
            x = TokenIndex._query_to_token_ids(
                q_vectors[i : i + max_batch_size], data.p, index=self.index, doc_ids=data.doc_ids
            )
            if token_ids is None:
                token_ids = x
            else:
                token_ids = torch.cat((token_ids, x), dim=0)
        return token_ids

    def __call__(
        self, q_vectors: torch.Tensor, p: int, doc_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        return self.process_data(FaissInput(q_vectors, p, doc_ids=doc_ids))

    def cleanup(self):
        del self._index

    @staticmethod
    @torch.no_grad()
    def _query_to_token_ids(
        Q: torch.Tensor,
        faiss_depth: int,
        *,
        index: IndexEngine,
        doc_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Query the faiss index for each embedding vector"""

        # flatten queries as [*, dim]
        num_queries, embeddings_per_query, dim = Q.shape
        Q = Q.view(-1, dim)

        # build the mask: queries with all vectors dimensions equal
        # to zero are considered to be padded
        zero_mask = Q.abs().sum(dim=-1) == 0

        # expand doc_ids
        if doc_ids is not None:
            doc_ids = [[i] * embeddings_per_query for i in doc_ids]
            doc_ids = [item for sublist in doc_ids for item in sublist]

        # todo: refactor
        max_bs = len(Q)
        token_ids = None
        for i in range(0, len(Q), max_bs):
            x = Q[i : i + max_bs]
            ids = doc_ids[i : i + max_bs] if doc_ids is not None else None
            _, y = index(x.to(torch.float32), k=faiss_depth, doc_ids=ids)

            # cast ids to Tensor and reshape as [bs, *]
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y)

            if token_ids is None:
                token_ids = y
            else:
                token_ids = torch.cat((token_ids, y), dim=0)

        # apply the mask
        token_ids[zero_mask, :] = -1

        # reshape
        token_ids = token_ids.view(num_queries, -1)
        return token_ids.to(Q.device)
