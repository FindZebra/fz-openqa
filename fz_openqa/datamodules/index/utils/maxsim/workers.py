from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple

import faiss
import numpy as np
import rich
import torch
from torch import LongTensor
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from fz_openqa.datamodules.index.utils.io import log_mem_size
from fz_openqa.datamodules.index.utils.io import read_vectors_from_table
from fz_openqa.datamodules.index.utils.maxsim.base_worker import format_device
from fz_openqa.datamodules.index.utils.maxsim.base_worker import TensorReducerWorker
from fz_openqa.datamodules.index.utils.maxsim.base_worker import TensorWorker
from fz_openqa.utils.datastruct import PathLike
from fz_openqa.utils.tensor_arrow import TensorArrowTable

logger = logging.getLogger(__name__)


@dataclass
class FaissInput:
    q_vectors: Tensor
    k: int
    p: int
    idx: int = None


@dataclass
class MaxSimInput:
    q_vectors: Tensor
    token_ids: Tensor
    k: int
    idx: int = None


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
    ) -> MaxSimOutput:
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
            chunksize = max(1, self.max_chunksize // q_vectors.shape[0])
        else:
            chunksize = pids.shape[1]
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
        return MaxSimOutput(scores=scores, pids=pids, k=k, boundaries=self.boundaries)

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


class MaxSimWorker(TensorWorker):
    """This class allows computing MaxSim for a subset of vectors (contained in MaxSimRanker)."""

    def __init__(
        self,
        max_sim: MaxSimRanker,
        id: int,
        **kwargs,
    ):
        super(MaxSimWorker, self).__init__(**kwargs)

        self.id = id
        self.max_sim = max_sim

    def _print(self, data):
        import rich  # noqa: F811

        rich.print(
            f"---- {type(self).__name__}(id={self.id}): {data}, device={self.max_sim.device}"
        )

    def cuda(self):
        assert isinstance(self.device, (int, torch.device))
        device = format_device(self.device)
        self.max_sim = self.max_sim.to(device)

    def cpu(self):
        self.max_sim = self.max_sim.to(torch.device("cpu"))

    def process_data(self, data):
        assert isinstance(data, MaxSimInput)
        output = self.max_sim(data.q_vectors, data.token_ids, data.k)
        assert isinstance(output, MaxSimOutput)
        self.output(output)
        del data

    def cleanup(self):
        del self.max_sim


class FaissWorker(TensorWorker):
    """This class handle a token-level faiss index"""

    def __init__(
        self,
        index: faiss.Index | PathLike,
        device: List[int],
        **kwargs,
    ):

        device = [i for i in set(device) if i >= 0]
        super(FaissWorker, self).__init__(device=device, **kwargs)
        assert isinstance(device, list)
        if isinstance(index, faiss.Index):
            self.index = faiss.serialize_index(index)
        else:
            self.index = str(index)

    def prepare(self):
        if isinstance(self.index, str):
            self.index = faiss.read_index(self.index)
        else:
            self.index = faiss.deserialize_index(self.index)

    def cuda(self):
        if len(self.device) > 0 and not isinstance(self.index, faiss.IndexReplicas):
            try:
                nprobe = self._index.nprobe
            except Exception:
                nprobe = None
            self.index = faiss.index_cpu_to_gpus_list(self.index, gpus=self.device)
            if nprobe is not None:
                params = faiss.GpuParameterSpace()  # type: ignore
                params.set_index_parameter(self.index, "nprobe", nprobe)

    def cpu(self):
        try:
            self._index = faiss.index_gpu_to_cpu(self._index)  # type: ignore
        except Exception:
            pass

    def process_data(self, data):
        assert isinstance(data, FaissInput)
        token_ids = FaissWorker._query_to_embedding_ids(data.q_vectors, data.k, index=self.index)
        output = MaxSimInput(q_vectors=data.q_vectors.clone(), token_ids=token_ids, k=data.k)
        self.output(output)
        del data

    def cleanup(self):
        del self.index

    @staticmethod
    @torch.no_grad()
    def _query_to_embedding_ids(Q: Tensor, faiss_depth: int, *, index: faiss.Index) -> Tensor:
        """Query the faiss index for each embedding vector"""
        num_queries, embeddings_per_query, dim = Q.shape
        Q = Q.view(-1, dim)
        _, embedding_ids = index.search(Q.to(torch.float32), faiss_depth)
        if isinstance(embedding_ids, np.ndarray):
            embedding_ids = torch.from_numpy(embedding_ids)
        embedding_ids = embedding_ids.view(num_queries, -1)
        return embedding_ids.to(Q.device)


class MaxSimReducerWorker(TensorReducerWorker):
    """Reduce the `MaxSimOutput`s from each `MaxSimWorker` to a single `MaxSimOutput`"""

    def prepare(self):
        pass

    def cpu(self):
        pass

    def cuda(self):
        pass

    def process_data(self, data: List[MaxSimOutput]):
        assert isinstance(data, list)
        assert all(isinstance(d, MaxSimOutput) for d in data)

        # gather data
        ks = [d.k for d in data]
        assert all(ks[0] == k for k in ks)
        k = ks[0]
        scores = [d.scores for d in data]
        pids = [d.pids for d in data]

        # send to device
        devices = list(set(s.device for s in scores))
        if len(devices) > 1:
            # when received tensors from different devices, randomly choose a device
            device = random.choice(devices)
            scores = [s.to(device, non_blocking=True) for s in scores]
            pids = [p.to(device, non_blocking=True) for p in pids]

        # concatenate
        all_scores, all_pids = (torch.cat(x, dim=-1) for x in (scores, pids))
        del pids, scores

        # take the top-k results given the MaxSim score
        k_ = min(k, all_scores.shape[-1])
        all_scores = all_scores.to(torch.float32)
        _, maxsim_idx = torch.topk(all_scores, k=k_, dim=-1, largest=True, sorted=True)

        # fetch the corresponding document indices and return
        maxsim_scores = all_scores.gather(index=maxsim_idx, dim=1)
        maxsim_pids = all_pids.gather(index=maxsim_idx, dim=1)
        if maxsim_scores.shape[1] < k or maxsim_pids.shape[1] < k:
            maxsim_pids, maxsim_scores = self._pad_outputs(k, maxsim_pids, maxsim_scores)
        output = MaxSimOutput(
            scores=maxsim_scores.clone().cpu(), pids=maxsim_pids.clone().cpu(), k=k, boundaries=None
        )
        del all_scores, all_pids, maxsim_idx, maxsim_scores, maxsim_pids
        return output

    def cleanup(self):
        pass

    @staticmethod
    def _pad_outputs(k: int, maxsim_pids: Tensor, maxsim_scores: Tensor):
        # pad maxsim_scores with nans
        maxsim_scores = MaxSimReducerWorker._pad_to_length(maxsim_scores, k, -torch.inf)
        # pad maxsim_pids with zeros
        maxsim_pids = MaxSimReducerWorker._pad_to_length(maxsim_pids, k, -1)
        return maxsim_pids, maxsim_scores

    @staticmethod
    def _pad_to_length(values: Tensor, k: int, fill_value=torch.nan):
        if values.shape[1] < k:
            return F.pad(values, (0, k - values.shape[1]), value=fill_value)
        else:
            return values
