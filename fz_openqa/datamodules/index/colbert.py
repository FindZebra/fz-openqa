from __future__ import annotations

import logging
import time
from typing import List
from typing import Optional

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import pyarrow as pa
import rich
import torch
import torch.nn.functional as F
from datasets import Split
from faiss import IndexReplicas
from torch import nn
from torch import Tensor

from fz_openqa.datamodules.index.dense import FaissIndex
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.datamodules.pipes import Predict
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import OutputFormat
from fz_openqa.utils.tensor_arrow import TensorArrowTable

# required to allow searching faiss with tensors

log = logging.getLogger(__name__)


def log_mem_size(x, msg, logger=None):
    if isinstance(x, torch.Tensor):
        mem_size = x.element_size() * x.nelement()
    elif isinstance(x, np.ndarray):
        mem_size = x.dtype.itemsize * x.size
    else:
        raise TypeError(f"Unsupported type {type(x)}")
    mem_size /= 1024 ** 2
    prec = "MB"
    if mem_size > 1024:
        mem_size /= 1024
        prec = "GB"
    msg = f"{msg} mem. size={mem_size:.3f} {prec}, shape={x.shape}, dtype={x.dtype}"
    if logger is None:
        rich.print(msg)
    else:
        logger.info(msg)


class ColbertIndex(FaissIndex):
    """
    Implementation of the Colbert index. This implementation supports multi-GPU.
    Half of the GPUs are allocated to faiss, and the other half to the MaxSim operator.

    Notes
    -----
    Future work:
    1. todo: Handle documents of different lengths (check the original repo
    2. todo: Handle sampling vectors from the TensorArrowTable without holding them in memory

    """

    _emb2pid: Optional[Tensor] = None
    _vectors: Optional[Tensor] = None
    _max_sim: Optional[MaxSim, nn.DataParallel] = None
    _is_gpu: bool = False
    _max_add_per_gpu = 1 << 25

    @property
    def vectors_table(self) -> TensorArrowTable:
        if self._vectors_table is None:
            self._vectors_table = self._read_vectors_table()
        return self._vectors_table

    @property
    def vectors(self) -> Tensor:
        if self._vectors is None:
            vectors_table = self.vectors_table
            self._vectors = self._read_vectors_from_table(vectors_table)
            log_mem_size(self._vectors, "Loaded Colbert vectors", logger=log)
        return self._vectors

    @property
    def emb2pid(self) -> Tensor:
        if self._emb2pid is None:
            self._emb2pid = self._build_emb2pid_from_vectors(self.vectors_table)
        return self._emb2pid

    @torch.no_grad()
    def _read_vectors_from_table(self, vectors_table: TensorArrowTable) -> Tensor:
        mem_size = vectors_table.nbytes // 1024 ** 3
        log.info(
            f"Reading {vectors_table.num_rows} vectors ({mem_size:.3f} GB) "
            f"from {vectors_table.path}"
        )
        start_time = time.time()
        vectors = vectors_table[0 : len(vectors_table)]
        log.info(
            f"Read {vectors_table.num_rows} vectors. Elapsed time: {time.time() - start_time:.3f}s"
        )
        return vectors

    @torch.no_grad()
    def _build_emb2pid_from_vectors(self, vectors_table: TensorArrowTable) -> Tensor:
        emb2pid = torch.arange(vectors_table.num_rows, dtype=torch.long)
        if self._vectors is not None:
            one_vec = self._vectors[0]
        else:
            one_vec = vectors_table[0]
        vector_length = one_vec.shape[0]
        emb2pid = emb2pid[:, None].expand(-1, vector_length)
        return emb2pid.reshape(-1).contiguous()

    @torch.no_grad()
    def _search_batch(
        self,
        query: Batch,
        *,
        idx: Optional[List[int]] = None,
        split: Optional[Split] = None,
        k: int = 1,
        faiss_batch_size: int = 2,
        **kwargs,
    ) -> SearchResult:

        # 1. get query vector
        query = self.predict_queries(query, format=OutputFormat.TORCH, idx=idx, split=split)
        q_vectors: Tensor = query[Predict.output_key]

        # 2. get faiss index, and potentially move to gpu
        n_gpus = faiss.get_num_gpus()
        faiss_gpus, maxsim_gpus = self._allocate_gpus(n_gpus)
        if len(faiss_gpus) > 0:
            if not isinstance(self._index, IndexReplicas):
                log.info(f"Moving faiss index to gpu {faiss_gpus}")
                nprobe = self._index.nprobe
                self._index = faiss.index_cpu_to_gpus_list(self._index, gpus=faiss_gpus)
                faiss.GpuParameterSpace().set_index_parameter(
                    self._index, "nprobe", nprobe
                )  # type: ignore

        # 3. build the MaxSim object
        if self._max_sim is None:
            self._max_sim = MaxSim(self.vectors, self.emb2pid)
            if len(maxsim_gpus) > 0:
                log.info(f"Moving MaxSim to gpu {maxsim_gpus}")
                self._max_sim = nn.DataParallel(self._max_sim, device_ids=maxsim_gpus)
                self._max_sim.to("cuda")

        # 4. query faiss (faiss index + MaxSim Dataparallel)
        eff_batch_size = faiss_batch_size * max(1, n_gpus)
        log.debug(f"Searching queries: {q_vectors.shape}, with batch size {eff_batch_size}")
        search_results = None
        for i in range(0, q_vectors.shape[0], eff_batch_size):
            q_vec = q_vectors[i : i + eff_batch_size]
            r = self._search_vectors(q_vec, k=k, index=self._index, max_sim=self._max_sim)

            if search_results is None:
                search_results = r
            else:
                search_results += r

        return search_results

    def _allocate_gpus(self, n_gpus):
        gpus = list(range(n_gpus))
        if n_gpus > 1:
            faiss_gpus = gpus[n_gpus // 2 :]
            maxsim_gpus = gpus[: n_gpus // 2]
        elif n_gpus == 1:
            faiss_gpus = gpus
            maxsim_gpus = []
        else:
            faiss_gpus = []
            maxsim_gpus = []
        return faiss_gpus, maxsim_gpus

    @torch.no_grad()
    def _search_vectors(
        self,
        q_vectors: Tensor,
        *,
        k: int,
        index: FaissIndex | IndexReplicas,
        max_sim: MaxSim,
        **kwargs,
    ) -> SearchResult:
        """
        Search the index using the `query` and
        return the index of the results within the original dataset.
        """
        start = time.time()

        # 2. query the token index
        # todo: allow setting max value
        p = min(20, max(1, k // 2))
        topk_indices = self._query_to_embedding_ids(q_vectors, p, index=index)
        faiss_time = time.time()
        log.debug(f"faiss={faiss_time - start:.3f}s, topk_indices={topk_indices.shape}")

        # apply max sim
        if isinstance(max_sim, nn.DataParallel):
            q_vectors = q_vectors.to("cuda")
            topk_indices = topk_indices.to("cuda")

        log.debug(f"Querying maxsim: {q_vectors.shape}, device={q_vectors.device}")
        batch = {"q_vectors": q_vectors, "topk_indices": topk_indices}
        max_sim_out = max_sim(batch, k=k)
        maxsim_pids, maxsim_scores = max_sim_out["pids"], max_sim_out["scores"]
        log.debug(f"max_sim={time.time() - faiss_time:.3f}s, topk_indices={topk_indices.shape}")

        log.debug(
            f"retrieval_time={time.time() - start:.3f}s, k={k}, maxsim_pids={maxsim_pids.shape}"
        )
        return SearchResult(
            score=maxsim_scores.cpu().numpy(),
            index=maxsim_pids.cpu().numpy(),
            dataset_size=self.dataset_size,
            k=k,
        )

    @torch.no_grad()
    def _retrieve_pid_vectors(
        self, pids: Tensor, *, vectors: pa.Table | np.ndarray | Tensor, dtype: torch.dtype
    ) -> Tensor:
        """Retrieve the vectors for the given pids."""
        if isinstance(vectors, pa.Table):
            batch_size, depth = pids.shape

            # flatten and get unique documents
            pids = pids.view(-1)
            unique_pids, indices = torch.unique(pids, return_inverse=True)
            unique_pids[unique_pids < 0] = 0

            # retrieve unique document vectors and cast as array
            unique_pid_vectors = self._vectors_table[unique_pids]

            # reshape and return
            pid_vectors = unique_pid_vectors[indices]
            return pid_vectors.view(batch_size, depth, *pid_vectors.shape[1:])
        elif isinstance(vectors, (np.ndarray, Tensor)):
            return vectors[pids]
        else:
            raise TypeError(f"vectors type {type(vectors)} not supported")

    @torch.no_grad()
    def _query_to_embedding_ids(self, Q: Tensor, faiss_depth: int, *, index: FaissIndex) -> Tensor:
        """Query the faiss index for each embedding vector"""
        num_queries, embeddings_per_query, dim = Q.shape
        Q = Q.reshape(-1, Q.shape[-1])
        _, embedding_ids = index.search(Q.to(torch.float32), faiss_depth)
        if isinstance(embedding_ids, np.ndarray):
            embedding_ids = torch.from_numpy(embedding_ids)
        embedding_ids = embedding_ids.view(num_queries, -1)
        return embedding_ids.to(Q.device)

    def __del__(self):
        if self._max_sim is not None:
            self._max_sim = self._max_sim.to("cpu")
            del self._max_sim
        super(ColbertIndex, self).__del__()


class MaxSim(nn.Module):
    def __init__(self, vectors: Tensor, emb2pid: Tensor):
        super(MaxSim, self).__init__()
        self.register_buffer("vectors", vectors)
        self.register_buffer("emb2pid", emb2pid)

    @torch.no_grad()
    def forward(self, batch: Batch = None, *, k: int, chunk_size: int = 1000) -> Batch:
        if batch is None:
            # todo: return emtpy tensor
            maxsim_pids = torch.empty((0, k), dtype=torch.long, device=self.vectors.device)
            maxsim_scores = torch.empty(
                (0, k), dtype=self.vectors.dtype, device=self.vectors.device
            )
            return {"pids": maxsim_pids, "scores": maxsim_scores}
        q_vectors: Tensor = batch["q_vectors"]
        topk_indices: Tensor = batch["topk_indices"]

        # 1. get the corresponding document ids
        pids = self.emb2pid[topk_indices]
        pids = self._get_unique_pids(pids)

        # 2. retrieve the vectors for each unique document index and compute the similarity
        scores = torch.zeros(
            q_vectors.shape[0], pids.shape[1], dtype=q_vectors.dtype, device=q_vectors.device
        )
        for i in range(0, pids.shape[1], chunk_size):
            scores[:, i : i + chunk_size] = self._score(pids[:, i : i + chunk_size], q_vectors)

        # 3. take the top-k results given the MaxSim score
        p = min(k, scores.shape[-1])
        scores = scores.to(torch.float32)
        _, maxsim_idx = torch.topk(scores, k=p, dim=-1, largest=True, sorted=True)
        # 4. fetch the corresponding document indices and return
        maxsim_scores = scores.gather(index=maxsim_idx, dim=1)
        maxsim_pids = pids.gather(index=maxsim_idx, dim=1)

        if maxsim_scores.shape[1] < k or maxsim_pids.shape[1] < k:
            # todo: check why this happens
            # logging.warning(f"MaxSim: k={k}, maxsim_scores={maxsim_scores.shape}, "
            #                 f"maxsim_pids={maxsim_pids.shape}")
            maxsim_pids, maxsim_scores = self._pad_outputs(k, maxsim_pids, maxsim_scores)
            # logging.warning(f"--> MaxSim: k={k}, maxsim_scores={maxsim_scores.shape}, "
            #                 f"maxsim_pids={maxsim_pids.shape}")

        return {"pids": maxsim_pids, "scores": maxsim_scores}

    def _pad_outputs(self, k, maxsim_pids, maxsim_scores):
        # pad maxsim_scores with nans
        missing_scores = torch.empty(
            maxsim_scores.shape[0],
            k - maxsim_scores.shape[1],
            device=maxsim_scores.device,
            dtype=maxsim_scores.dtype,
        )
        missing_scores.fill_(torch.nan)
        maxsim_scores = torch.cat([maxsim_scores, missing_scores], dim=1)
        # pad maxsim_pids with zeros
        missing_pids = torch.zeros(
            maxsim_pids.shape[0],
            k - maxsim_pids.shape[1],
            device=maxsim_pids.device,
            dtype=maxsim_pids.dtype,
        )
        maxsim_pids = torch.cat([maxsim_pids, missing_pids], dim=1)
        return maxsim_pids, maxsim_scores

    def _score(self, pids, q_vectors):
        d_vectors = self._retrieve_pid_vectors(pids)
        # apply max sim to the retrieved vectors
        scores = torch.einsum("bqh, bkdh -> bkqd", q_vectors, d_vectors)
        # max. over the documents tokens, for each query token
        scores, _ = scores.max(axis=-1)
        # avg over all query tokens
        scores = scores.mean(axis=-1)
        # set the score to -inf for the negative pids
        scores[pids < 0] = -torch.inf

        return scores

    def _retrieve_pid_vectors(self, pids: Tensor) -> Tensor:
        return self.vectors[pids]

    def _get_unique_pids(self, pids: Tensor, fill_value=-1) -> Tensor:
        upids = [torch.unique(r) for r in pids]
        max_length = max(len(r) for r in upids)

        def _pad(r):
            return F.pad(r, (0, max_length - len(r)), value=fill_value)

        return torch.cat([_pad(p)[None] for p in upids])
