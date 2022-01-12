from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
import rich
import torch
from torch import Tensor
from torch.nn import functional as F

from fz_openqa.datamodules.index.utils.maxsim.base_worker import format_device
from fz_openqa.datamodules.index.utils.maxsim.base_worker import TensorReducerWorker
from fz_openqa.datamodules.index.utils.maxsim.base_worker import TensorWorker
from fz_openqa.datamodules.index.utils.maxsim.ranker import MaxSimOutput
from fz_openqa.datamodules.index.utils.maxsim.ranker import MaxSimRanker
from fz_openqa.utils.datastruct import PathLike

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
        self.max_sim = self.max_sim.to(device, non_blocking=False)

    def cpu(self):
        self.max_sim = self.max_sim.to(torch.device("cpu"), non_blocking=False)

    def process_data(self, data):
        # rich.print(f"> {type(self).__name__}(id={self.id})")
        assert isinstance(data, MaxSimInput)
        scores, pids = self.max_sim(data.q_vectors, data.token_ids, data.k)
        output = MaxSimOutput(
            scores=scores, pids=pids, k=data.k, boundaries=self.max_sim.boundaries
        )
        self.output(output)
        # del data.q_vectors, data.token_ids, data.k, data.idx

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
        # rich.print(f"> {type(self).__name__}(id={self.id})")
        assert isinstance(data, FaissInput)
        token_ids = FaissWorker._query_to_embedding_ids(data.q_vectors, data.k, index=self.index)
        output = MaxSimInput(q_vectors=data.q_vectors, token_ids=token_ids, k=data.k)
        self.output(output)
        # del data # todo

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
        # rich.print(f"> {type(self).__name__}(id={self.id})")

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
        # del pids, scores

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
        # del all_scores, all_pids, maxsim_idx, maxsim_scores, maxsim_pids
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
