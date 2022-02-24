from __future__ import annotations

import logging
from typing import Optional
from typing import Tuple

import rich
import torch
from torch import LongTensor
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from fz_openqa.datamodules.index.utils.io import log_mem_size
from fz_openqa.datamodules.index.utils.io import read_vectors_from_table
from fz_openqa.utils.tensor_arrow import TensorArrowTable

logger = logging.getLogger(__name__)


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

        # get the unique pids
        pids = self._get_unique_pids(pids)

        # compute the scores
        scores = torch.zeros(
            q_vectors.shape[0], pids.shape[1], dtype=q_vectors.dtype, device=q_vectors.device
        )

        if self.max_chunksize is not None:
            # the chunksize must be divided by the batch size of q_vectors
            chunksize = max(1, self.max_chunksize // q_vectors.shape[0])
            chunksize = min(chunksize, pids.shape[1])
        else:
            chunksize = pids.shape[1]

        for i in range(0, pids.shape[1], chunksize):
            pid_chunk = pids[:, i : i + chunksize]
            scores[:, i : i + chunksize] = self._score(pid_chunk, q_vectors, self.vectors)

        # if k is specified, return the top k
        if k is not None and k < scores.shape[1]:
            _, maxsim_idx = torch.topk(scores, k=k, dim=-1, largest=True, sorted=True)

            scores = scores.gather(index=maxsim_idx, dim=1)
            pids = pids.gather(index=maxsim_idx, dim=1)

        # recover pid offset, and return
        pids[pids >= 0] += self.boundaries[0]
        # rich.print(f">> ranker.pids: {pids.shape}, scores:{scores.shape}")
        return scores, pids

    @staticmethod
    def _get_unique_pids(pids: Tensor, fill_value=-1) -> Tensor:
        """
        Get the unique pids across dimension 1 and pad to the max length.
        `torch.unique` sorts the pids in ascending order, so we need to reverse with -1"""
        upids = [torch.unique(r) for r in torch.unbind(-pids)]
        max_length = max(len(r) for r in upids)

        def _pad(r):
            return F.pad(r, (0, max_length - len(r)), value=fill_value)

        return -torch.cat([_pad(p)[None] for p in upids])

    @staticmethod
    def _score(pids: LongTensor, q_vectors: Tensor, vectors: Tensor):
        d_vectors = vectors[pids]

        # apply max sim to the retrieved vectors
        scores = torch.einsum("bqh, bkdh -> bkqd", q_vectors, d_vectors)
        # max. over the documents tokens, for each query token
        scores, _ = scores.max(axis=-1)
        # avg over all query tokens (length dimension)
        scores = scores.mean(axis=-1)
        # set the score to -inf for the negative pids
        scores[pids < 0] = -torch.inf

        return scores

    def __del__(self):
        del self.vectors
        del self.boundaries
