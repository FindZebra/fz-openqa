from __future__ import annotations

import logging
from dataclasses import dataclass
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


@dataclass
class MaxSimOutput:
    pids: Tensor
    scores: Tensor
    boundaries: Optional[Tuple[int, int]]
    k: int
    idx: int = None


class MaxSimRanker(nn.Module):
    """Compute MaxSim for a subset of vectors"""

    def __init__(
        self,
        emb2pid: Tensor | TensorArrowTable,
        vectors: Tensor | TensorArrowTable,
        boundaries: Tuple[int, int],
        max_chunksize: Optional[int] = None,
    ):
        super(MaxSimRanker, self).__init__()
        # todo: apply boundaries to emb2pid
        emb2pid = read_vectors_from_table(emb2pid)
        self.register_buffer("emb2pid", emb2pid)
        vectors = read_vectors_from_table(vectors, boundaries=boundaries)
        self.register_buffer("vectors", vectors)
        log_mem_size(self.vectors, "MaxSimRanker vectors", logger=logger)
        self.max_chunksize = max_chunksize
        self.boundaries = boundaries
        assert len(self.vectors) == boundaries[1] - boundaries[0]

    @property
    def device(self) -> torch.device:
        return self.vectors.device

    @torch.no_grad()
    def forward(
        self, q_vectors: Tensor, token_ids: Tensor, k: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """Compute the max similarity for a batch of query vectors token_ids."""
        # if not q_vectors.device == self.device:
        #     q_vectors = q_vectors.to(self.device)
        # if not token_ids.device == self.device:
        #     token_ids = token_ids.to(self.device)

        # get the document vectors
        pids = self.emb2pid[token_ids]

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
            # the chubnksize must be divided by the batch size of q_vectors
            chunksize = max(1, self.max_chunksize // q_vectors.shape[0])
        else:
            chunksize = pids.shape[1]

        # rich.print(f">> ranker.chunksize: {chunksize}")
        for i in range(0, pids.shape[1], chunksize):
            scores[:, i : i + chunksize] = self._score(
                pids[:, i : i + chunksize], q_vectors, self.vectors
            )

        # if k is specified, return the top k
        if k is not None and k < scores.shape[1]:
            _, maxsim_idx = torch.topk(scores, k=k, dim=-1, largest=True, sorted=True)

            scores = scores.gather(index=maxsim_idx, dim=1)
            pids = pids.gather(index=maxsim_idx, dim=1)

        # recover pid offset, and return
        pids += self.boundaries[0]
        # rich.print(f">> ranker.pids: {pids.shape}, scores:{scores.shape}")
        return scores, pids

    @staticmethod
    def _get_unique_pids(pids: Tensor, fill_value=-1) -> Tensor:
        """
        Get the unique pids across dimension 1 and pad to the max length.
        `torch.unique` sorts the pids, we use descending sort `* -1` to
        place negative numbers last."""
        upids = [-torch.unique(r) for r in torch.unbind(-pids)]
        max_length = max(len(r) for r in upids)

        def _pad(r):
            return F.pad(r, (0, max_length - len(r)), value=fill_value)

        return torch.cat([_pad(p)[None] for p in upids])

    @staticmethod
    def _score(pids: LongTensor, q_vectors: Tensor, vectors: Tensor):
        d_vectors = vectors[pids]

        # apply max sim to the retrieved vectors
        scores = torch.einsum("bqh, bkdh -> bkqd", q_vectors, d_vectors)
        # max. over the documents tokens, for each query token
        scores, _ = scores.max(axis=-1)
        # avg over all query tokens
        scores = scores.sum(axis=-1)
        # set the score to -inf for the negative pids
        scores[pids < 0] = -torch.inf

        return scores

    def __del__(self):
        del self.vectors
        del self.boundaries
