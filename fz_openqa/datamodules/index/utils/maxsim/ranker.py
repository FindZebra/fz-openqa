from __future__ import annotations

from typing import Optional
from typing import Tuple

import torch
from loguru import logger
from torch import LongTensor
from torch import nn
from torch import Tensor

from fz_openqa.datamodules.index.utils.io import log_mem_size
from fz_openqa.datamodules.index.utils.io import read_vectors_from_table
from fz_openqa.datamodules.index.utils.maxsim.utils import get_unique_pids
from fz_openqa.utils.tensor_arrow import TensorArrowTable


class MaxSimRanker(nn.Module):
    """Compute MaxSim for a subset of vectors"""

    def __init__(
        self,
        vectors: Tensor | TensorArrowTable,
        boundaries: Tuple[int, int] | Tensor,
        max_chunksize: Optional[int] = None,
    ):
        super(MaxSimRanker, self).__init__()
        vectors = read_vectors_from_table(vectors, boundaries=boundaries)
        self.register_buffer("vectors", vectors)
        log_mem_size(self.vectors, "MaxSimRanker vectors", logger=logger)
        self.max_chunksize = max_chunksize

        # register boundaries
        if boundaries is None:
            boundaries = (0, vectors.shape[0])
        if not isinstance(boundaries, Tensor):
            boundaries = torch.tensor([boundaries[0], boundaries[1]], dtype=torch.long)
        assert boundaries.shape == (2,)
        self.register_buffer("boundaries", boundaries)
        if len(self.vectors) != boundaries[1] - boundaries[0]:
            raise ValueError(
                f"boundaries do not match vectors: {boundaries}. "
                f"Vector length: {len(self.vectors)}, "
                f"does not chunk size: {boundaries[1] - boundaries[0]}"
            )

    @property
    def device(self) -> torch.device:
        return self.vectors.device

    @torch.no_grad()
    def forward(
        self, q_vectors: Tensor, pids: Tensor, k: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """Compute the max similarity for a batch of query vectors token_ids."""
        # offset pids
        pids -= self.boundaries[0]

        # apply the offset and set all pids that are out of range to -1
        pids[(pids < 0) | (pids >= len(self.vectors))] = -1

        # get the unique pids across the batch dimension
        pids = get_unique_pids(pids)

        # compute the scores
        scores = torch.empty(
            q_vectors.shape[0], pids.shape[1], dtype=q_vectors.dtype, device=q_vectors.device
        ).fill_(-torch.inf)

        if self.max_chunksize is not None:
            # the chunksize must be divided by the batch size of q_vectors
            chunksize = max(1, self.max_chunksize // q_vectors.shape[0])
            chunksize = min(chunksize, pids.shape[1])
        else:
            chunksize = pids.shape[1]

        for i in range(0, pids.shape[1], chunksize):
            pid_chunk = pids[:, i : i + chunksize]
            scores_chunk = self._score(pid_chunk, q_vectors, self.vectors)
            scores[:, i : i + chunksize] = scores_chunk

        # if k is unspecified, only sort
        if k is None or k > scores.shape[1]:
            k = scores.shape[1]

        # retriever the top-k documents
        maxsim_idx = torch.topk(scores, k=k, dim=-1, largest=True, sorted=True).indices
        scores = scores.gather(index=maxsim_idx, dim=-1)
        pids = pids.gather(index=maxsim_idx, dim=-1)

        # recover pid offset, and return
        pids[pids >= 0] += self.boundaries[0]
        return scores, pids

    @staticmethod
    def _score(pids: LongTensor, q_vectors: Tensor, vectors: Tensor):
        d_vectors = vectors[pids]

        # apply max sim to the retrieved vectors
        scores = torch.einsum("bqh, bkdh -> bkqd", q_vectors, d_vectors)
        # max. over the documents tokens, for each query token
        scores = scores.max(axis=-1).values
        # avg over all query tokens (length dimension)
        scores = scores.sum(axis=-1)
        # set the score to -inf for the negative pids
        scores[pids < 0] = -torch.inf

        return scores

    def __del__(self):
        del self.vectors
        del self.boundaries
