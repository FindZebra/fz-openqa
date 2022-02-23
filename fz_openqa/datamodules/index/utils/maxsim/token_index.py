from __future__ import annotations

from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import rich
import torch

from fz_openqa.datamodules.index.handlers.base import IndexHandler
from fz_openqa.datamodules.index.utils.maxsim.datastruct import FaissInput
from fz_openqa.utils.datastruct import PathLike


class TokenIndex(object):
    """This class handle a token-level faiss index"""

    def __init__(
        self,
        index: IndexHandler | PathLike,
        devices: List[int],
        multiprocessing: bool = False,
        **kwargs,
    ):
        self.devices = [i for i in set(devices) if i >= 0]
        super(TokenIndex, self).__init__()
        assert isinstance(devices, list)

        if not isinstance(index, IndexHandler):
            assert isinstance(index, (str, Path))
        self._index: IndexHandler | PathLike = index

        if not multiprocessing:
            self.prepare()

    def prepare(self):
        if isinstance(self._index, (str, Path)):
            self._index = IndexHandler.load_from_path(self._index)
            self._index.load()

    @property
    def index(self) -> IndexHandler:
        if not isinstance(self._index, IndexHandler):
            self.prepare()
        return self._index

    def cuda(self):
        self.index.cuda(devices=self.devices)

    def cpu(self):
        self.index.cpu()

    def process_data(self, data):
        # rich.print(f"> {type(self).__name__}(id={self.id})")
        assert isinstance(data, FaissInput)
        token_ids = TokenIndex._query_to_embedding_ids(
            data.q_vectors, data.p, index=self.index, doc_ids=data.doc_ids
        )
        return token_ids

    def __call__(
        self, q_vectors: torch.Tensor, p: int, doc_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        return self.process_data(FaissInput(q_vectors, p, doc_ids=doc_ids))

    def cleanup(self):
        del self._index

    @staticmethod
    @torch.no_grad()
    def _query_to_embedding_ids(
        Q: torch.Tensor,
        faiss_depth: int,
        *,
        index: IndexHandler,
        doc_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Query the faiss index for each embedding vector"""

        # flatten queries as [*, dim]
        num_queries, embeddings_per_query, dim = Q.shape
        Q = Q.view(-1, dim)

        # expand doc_ids
        if doc_ids is not None:
            doc_ids = [[i] * embeddings_per_query for i in doc_ids]
            doc_ids = [item for sublist in doc_ids for item in sublist]

        _, embedding_ids = index(Q.to(torch.float32), k=faiss_depth, doc_ids=doc_ids)

        # cast ids to Tensor and reshape as [bs, *]
        if isinstance(embedding_ids, np.ndarray):
            embedding_ids = torch.from_numpy(embedding_ids)
        embedding_ids = embedding_ids.view(num_queries, -1)
        return embedding_ids.to(Q.device)
