from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import torch  # type: ignore
import torch.nn.functional as F
from datasets import Dataset
from loguru import logger

from fz_openqa.datamodules.index.engines.base import IndexEngine
from fz_openqa.datamodules.index.engines.faiss import FaissEngine
from fz_openqa.datamodules.index.engines.vector_base.utils.faiss import TensorLike
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.datamodules.index.utils.io import build_emb2pid_from_vectors
from fz_openqa.utils.tensor_arrow import TensorArrowTable


def _pad(r, max_length, fill_value):
    return F.pad(r, (0, max_length - len(r)), value=fill_value)


class FaissTokenEngine(FaissEngine):
    _max_num_proc: int = 1
    no_fingerprint: List[str] = FaissEngine.no_fingerprint + ["emb2pid"]
    _default_config: Dict[str, Any] = {
        "p": 10,
        "max_bs": 1 << 10,
        **FaissEngine._default_config,
    }

    def _build(
        self,
        vectors: Optional[TensorLike | TensorArrowTable] = None,
        corpus: Optional[Dataset] = None,
    ):
        # build the token to passage id lookup table
        self.emb2pid = build_emb2pid_from_vectors(vectors)

        # Store `emb2pid`
        self._validate_emb2pid(self.emb2pid, vectors)
        # Add `-1` : padding, so indexing with -1 returns -1
        ones = torch.ones_like(self.emb2pid[..., -1:])
        self.emb2pid = torch.cat([self.emb2pid, -ones], dim=-1)

        # flatten the vectors
        flat_vectors = vectors.view(-1, vectors.shape[-1])
        super(FaissTokenEngine, self)._build(vectors=flat_vectors, corpus=corpus)

    @staticmethod
    def _validate_emb2pid(emb2pid, vectors):
        if set(range(len(vectors))) != set(emb2pid.unique().tolist()):
            raise ValueError(
                f"All positions in `vector` must be in `emb2pid`."
                f"{set(range(len(vectors)))} != {set(emb2pid.unique().tolist())}"
            )

        if emb2pid.min() != 0:
            raise ValueError(f"`emb2pid` must start at 0. Found {emb2pid.min()}")

        if emb2pid.max() != len(vectors) - 1:
            raise ValueError(f"`emb2pid` must end at {len(vectors) - 1}. Found {emb2pid.max()}")

    def search(self, *query: Any, k: int = None, **kwargs) -> SearchResult:
        q_vectors, *_ = query

        # query the token index
        _time = time.time()
        scores, indices = self._query_to_token_ids(
            q_vectors, self.config["p"], index=self._index, max_bs=self.config["max_bs"]
        )
        faiss_time = time.time() - _time

        # retrieve the passage ids from the token ids
        pids = self.emb2pid[indices.to(self.emb2pid.device)]
        lookup_time = time.time() - faiss_time

        # Deduplicate pids; this step is done on `device`, which is set by default on CPU.
        # This is done with a for loop across batch size, which can be quite slow.
        scores, pids = self._deduplicate_pids(scores, pids)
        deduplicate_time = time.time() - lookup_time

        pad_ratio = (pids <= 0).float().sum() / pids.numel()
        logger.info(
            f"Faiss time: {faiss_time:.2f}s, "
            f"lookup time: {lookup_time:.2f}s, "
            f"deduplicate time: {deduplicate_time:.2f}s, "
            f"pids: {pids.shape}, "
            f"pad_ratio={pad_ratio:.2%}"
        )

        if scores.shape[1] > self.k:
            logger.warning(
                f"{type(self).__name__} search: "
                f"scores.shape[1]={scores.shape[1]} > self.k={self.k}. "
                f"Truncating scores to {self.k}."
            )
            scores = scores[:, : self.k]
            pids = pids[:, : self.k]

        return scores, pids

    def _deduplicate_pids(
        self, scores: torch.Tensor, pids: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        # todo: test this

        unique_pids = []
        sum_scores = []
        for qpids, qscores in zip(pids, scores):
            # ge the unique pids
            q_unique_pids, upids_inv = torch.unique(qpids, return_inverse=True)

            # sum the token scores for each pid
            q_sum_scores = torch.zeros_like(q_unique_pids, dtype=qscores.dtype)
            q_sum_scores.index_add_(0, upids_inv, qscores)

            # sort the pids and scores according to the scores
            idx = torch.argsort(q_sum_scores, descending=True)
            q_unique_pids = q_unique_pids[idx]
            q_sum_scores = q_sum_scores[idx]

            # store the results
            unique_pids.append(q_unique_pids)
            sum_scores.append(q_sum_scores)

        # pad and stack outputs
        max_length = max(map(len, unique_pids))
        unique_pids = torch.stack([_pad(p, max_length, fill_value=-1) for p in unique_pids])
        sum_scores = torch.stack([_pad(s, max_length, fill_value=-math.inf) for s in sum_scores])

        return sum_scores, unique_pids

    @property
    def emb2pid_path(self) -> Path:
        path = Path(self.path)
        return path / "emb2pid.pt"

    def save(self):
        super(FaissTokenEngine, self).save()
        torch.save(self.emb2pid, self.emb2pid_path.as_posix())

    def load(self):
        super(FaissTokenEngine, self).load()
        self.emb2pid = torch.load(self.emb2pid_path.as_posix())

    def free_memory(self):
        super(FaissTokenEngine, self).free_memory()
        self.emb2pid = None

    @property
    def is_up(self) -> bool:
        return super(FaissTokenEngine, self).is_up and self.emb2pid is not None

    @staticmethod
    @torch.no_grad()
    def _query_to_token_ids(
        Q: torch.Tensor,
        faiss_depth: int,
        *,
        index: IndexEngine,
        max_bs: int = None,
    ) -> (torch.Tensor, torch.Tensor):
        """Query the faiss index for each embedding vector"""

        # flatten queries as [*, dim]
        num_queries, embeddings_per_query, dim = Q.shape
        Q = Q.view(-1, dim)

        # build the mask: queries with all vectors dimensions equal
        # to zero are considered to be padded
        zero_mask = Q.abs().sum(dim=-1) == 0

        max_bs = max_bs or len(Q)
        token_ids = torch.empty(len(Q), faiss_depth, dtype=torch.long)
        token_scores = torch.empty(len(Q), faiss_depth, dtype=torch.float)
        for i in range(0, len(Q), max_bs):
            x = Q[i : i + max_bs]
            y_score, y_index = index.search(x.to(torch.float32), k=faiss_depth)

            # cast ids to Tensor and reshape as [bs, *]
            if isinstance(y_index, np.ndarray):
                y_index = torch.from_numpy(y_index)
            if isinstance(y_score, np.ndarray):
                y_score = torch.from_numpy(y_score)

            token_ids[i : i + max_bs] = y_index
            token_scores[i : i + max_bs] = y_score

        # apply the mask
        token_ids[zero_mask, :] = -1
        token_scores[zero_mask, :] = -math.inf

        # reshape
        token_ids = token_ids.view(num_queries, -1)
        token_scores = token_scores.view(num_queries, -1)
        return token_scores.to(Q.device), token_ids.to(Q.device)
