from __future__ import annotations

import logging
import time
from typing import List
from typing import Optional
from typing import Tuple

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import pyarrow as pa
import rich
import torch
import torch.nn.functional as F
from datasets import Split
from torch import nn
from torch import Tensor

from fz_openqa.datamodules.index.dense import FaissIndex
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.datamodules.pipes.predict import OutputFormat
from fz_openqa.utils.datastruct import Batch

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
    Implementiation of the Colbert index.

    Notes
    -----
    Assumes the documents to be of the same length!

    """

    _emb2pid: Optional[Tensor] = None
    _vectors: Optional[Tensor] = None
    _max_sim: Optional[MaxSim, nn.DataParallel] = None
    _is_gpu: bool = False

    @property
    def vectors_table(self) -> pa.Table:
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
    def _read_vectors_from_table(self, vectors_table: pa.Table, batch_size=1000) -> Tensor:
        log.info(f"Reading {vectors_table.num_rows} vectors")
        i = 0
        vectors = None
        for vectors_chunk in self._iter_vectors(
            vectors_table,
            batch_size=batch_size,
            progress_bar=self.progress_bar,
            output_format=OutputFormat.TORCH,
            pin_memory=True,
        ):
            if vectors is None:
                vectors = torch.zeros(
                    (vectors_table.num_rows, *vectors_chunk.shape[1:]), dtype=torch.float32
                )

            vectors[i : i + vectors_chunk.shape[0]] = vectors_chunk
            i += vectors_chunk.shape[0]

        return vectors

    @torch.no_grad()
    def _build_emb2pid_from_vectors(self, vectors_table: pa.Table) -> Tensor:
        emb2pid = np.arange(vectors_table.num_rows, dtype=np.long)
        if self._vectors is not None:
            one_vec = self._vectors[0]
        else:
            one_vec = next(iter(self._iter_vectors(vectors_table, batch_size=1, num_workers=0)))[0]
        vector_length = one_vec.shape[0]
        emb2pid = np.repeat(emb2pid[:, None], vector_length, axis=1)
        return torch.from_numpy(emb2pid.reshape(-1)).contiguous()

    @torch.no_grad()
    def _search_batch(
        self,
        query: Batch,
        *,
        idx: Optional[List[int]] = None,
        split: Optional[Split] = None,
        k: int = 1,
        **kwargs,
    ) -> SearchResult:
        """
        Search the index using the `query` and
        return the index of the results within the original dataset.
        # todo: inner batch size
        """

        # 0. load vectors
        if self.in_memory:
            if self._max_sim is None:
                self._max_sim = MaxSim(self.vectors, self.emb2pid)
        else:
            # documents = self.vectors_table
            # emb2pid = self.emb2pid
            raise NotImplementedError

        # place faiss index to GPUs

        ngpus = faiss.get_num_gpus()
        if ngpus > 0:
            gpus = list(range(ngpus))
            if not self._is_gpu:
                nprobe = self._index.nprobe
                self._index = faiss.index_cpu_to_gpus_list(self._index, gpus=gpus[ngpus // 2 :])
                self._index.nprobe = nprobe
                self._is_gpu = True
            if not isinstance(self._max_sim, nn.DataParallel):
                self._max_sim = nn.DataParallel(self._max_sim, device_ids=gpus[: ngpus // 2])
                self._max_sim.to("cuda")

        index = self._index

        # 1. get query vector
        query = self.predict_queries(query, format=OutputFormat.TORCH, idx=idx, split=split)
        query = self.postprocess(query)
        q_vectors: Tensor = query[self.vectors_column_name]
        start = time.time()
        log.info(f"Searching queries: {q_vectors.shape}")

        # 2. query the token index
        p = min(10, max(1, k // 2))
        topk_indices = self._query_to_embedding_ids(q_vectors, p, index=index)
        log.info(f"faiss={time.time() - start:.3f}s, topk_indices={topk_indices.shape}")

        # apply max sim
        if ngpus > 0:
            q_vectors = q_vectors.to("cuda")
            topk_indices = topk_indices.to("cuda")
        maxsim_pids, maxsim_scores = self._max_sim(q_vectors, topk_indices, k=k)

        log.info(f"retrieval_time={time.time() - start:.3f}s, k={k}")
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
            unique_pid_vectors = self._vectors_table.take(unique_pids.numpy()).to_pydict()
            unique_pid_vectors = self.postprocess(unique_pid_vectors)[self.vectors_column_name]
            unique_pid_vectors = torch.tensor(unique_pid_vectors, dtype=dtype)

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
        _, embedding_ids = index.search(Q, faiss_depth)
        if isinstance(embedding_ids, np.ndarray):
            embedding_ids = torch.from_numpy(embedding_ids)
        embedding_ids = embedding_ids.view(num_queries, -1)
        return embedding_ids.to(Q.device)


class MaxSim(nn.Module):
    # todo: don't scatter vectors and emd2pids to device: too cumbersome, use lower level data

    def __init__(self, vectors: Tensor, emb2pid: Tensor):
        super(MaxSim, self).__init__()
        self.register_buffer("vectors", vectors)
        self.register_buffer("emb2pid", emb2pid)

    @torch.no_grad()
    def forward(self, q_vectors: Tensor, topk_indices: Tensor, k: int):
        # 3. get the corresponding document ids
        pids = self.emb2pid[topk_indices]
        pids = self._get_unique_pids(pids)

        # 4. retrieve the vectors for each unique document index
        d_vectors = self._retrieve_pid_vectors(pids)
        # log.info(f"retrieved d_vectors={time.time() - start:.3f}s, d_vectors={d_vectors.shape}")
        # log_mem_size(d_vectors, "document vectors", logger=log)

        # 7. apply max sim to the retrieved vectors
        scores = torch.einsum("bqh, bkdh -> bkqd", q_vectors, d_vectors)
        # max. over the documents tokens, for each query token
        scores, _ = scores.max(axis=-1)
        # avg over all query tokens
        scores = scores.mean(axis=-1)
        # set the score to -inf for the negative pids
        scores[pids < 0] = -torch.inf

        # 8. take the top-k results given the MaxSim score
        k = min(k, scores.shape[-1])
        _, maxsim_idx = torch.topk(scores, k=k, dim=-1, largest=True, sorted=True)
        # 9. fetch the corresponding document indices and return
        maxsim_scores = scores.gather(index=maxsim_idx, dim=1)
        maxsim_pids = pids.gather(index=maxsim_idx, dim=1)
        return maxsim_pids, maxsim_scores

    def _retrieve_pid_vectors(self, pids: Tensor) -> Tensor:
        return self.vectors[pids]

    def _get_unique_pids(self, pids: Tensor, fill_value=-1) -> Tensor:
        upids = [torch.unique(r) for r in pids]
        max_length = max(len(r) for r in upids)

        def _pad(r):
            return F.pad(r, (0, max_length - len(r)), value=fill_value)

        return torch.cat([_pad(p)[None] for p in upids])
