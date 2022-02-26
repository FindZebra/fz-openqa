from __future__ import annotations

import logging
import time
from typing import List
from typing import Optional

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import pyarrow as pa
import torch
from datasets import Split
from faiss import IndexReplicas
from torch import Tensor

from fz_openqa.datamodules.index.dense import FaissIndex
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.datamodules.index.utils.maxsim import log_mem_size
from fz_openqa.datamodules.index.utils.maxsim import MaxSimParallel
from fz_openqa.datamodules.pipes import Predict
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import OutputFormat
from fz_openqa.utils.tensor_arrow import TensorArrowTable

log = logging.getLogger(__name__)


class ColbertIndex(FaissIndex):
    """
    Implementation of the Colbert index. This implementation supports multi-GPU.
    Half of the GPUs are allocated to faiss, and the other half to the MaxSim operator.

    Notes
    -----
    Future work:
    0. todo: cleanup the code
    1. todo: Handle documents of different lengths (check the original repo
    2. todo: Handle sampling vectors from the TensorArrowTable without holding them in memory

    """

    _emb2pid: Optional[Tensor] = None
    _vectors: Optional[Tensor] = None
    _max_sim: Optional[MaxSimParallel] = None
    _is_gpu: bool = False
    _max_add_per_gpu = 1 << 25
    _max_num_proc: int = 1

    no_fingerprint: List[str] = FaissIndex.no_fingerprint + [
        "_max_sim",
        "_vectors" "_emb2pid",
        "in_memory",
        "_is_gpu",
        "_max_add_per_gpu",
        "keep_maxsim_on_cpu",
        "maxsim_chunksize",
    ]

    def __init__(
        self,
        *args,
        p: int = 100,
        keep_maxsim_on_cpu: bool = False,
        maxsim_chunksize: int = 10000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.p = p
        self.keep_maxsim_on_cpu = keep_maxsim_on_cpu
        self.maxsim_chunksize = maxsim_chunksize

    def _init_index_and_move_to_gpus(self):
        """Initialize the FAISS index and the MaxSim object and move to gpus."""
        n_gpus = faiss.get_num_gpus()
        faiss_gpus, maxsim_gpus = self._allocate_gpus(n_gpus)
        if len(faiss_gpus) > 0:
            if not isinstance(self._index, IndexReplicas):
                log.info(f"Moving faiss index to gpu {faiss_gpus}")
                try:
                    nprobe = self._index.nprobe
                except Exception:
                    nprobe = None
                self._index = faiss.index_cpu_to_gpus_list(self._index, gpus=faiss_gpus)
                if nprobe is not None:
                    faiss.GpuParameterSpace().set_index_parameter(
                        self._index, "nprobe", nprobe
                    )  # type: ignore

        if self._max_sim is None:
            if len(maxsim_gpus) > 0:
                log.info(f"Moving MaxSim to gpu {maxsim_gpus}")
            self._max_sim = MaxSimParallel(
                self.vectors,
                self.emb2pid,
                device_ids=maxsim_gpus,
                max_chunksize=self.maxsim_chunksize,
            )

    def _preprocess_query(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Optional[Split] = None, **kwargs
    ) -> Batch:
        """Preprocess the batch before query"""
        return self.predict_queries(batch, idx=idx, split=split, format=OutputFormat.TORCH)

    @torch.no_grad()
    def _search_chunk(
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
        """
        self._init_index_and_move_to_gpus()
        start = time.time()
        q_vectors: Tensor = query[Predict.output_key]

        # 2. query the token index
        topk_indices = self._query_to_embedding_ids(q_vectors, self.p, index=self._index)
        faiss_time = time.time()
        log.debug(f"faiss={faiss_time - start:.3f}s, topk_indices={topk_indices.shape}")

        # apply max sim
        q_vectors = q_vectors.to(self._max_sim.devices[0], non_blocking=True)
        topk_indices = topk_indices.to(self._max_sim.devices[0], non_blocking=True)

        log.debug(f"Querying maxsim: {q_vectors.shape}, device={q_vectors.device}")
        batch = {"q_vectors": q_vectors, "topk_indices": topk_indices}
        max_sim_out = self._max_sim(batch, k=k)
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

    def to_cpu(self):
        super().to_cpu()
        if self._max_sim is not None:
            del self._max_sim
            self._max_sim = None

    def __del__(self):
        self.to_cpu()
        super(ColbertIndex, self).__del__()

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

    def _allocate_gpus(self, n_gpus):
        """Allocate GPUs to the faiss index and to max_sim"""
        gpus = list(range(n_gpus))
        if n_gpus > 1:
            if not self.keep_maxsim_on_cpu:
                faiss_gpus = gpus[n_gpus // 2 :]
                maxsim_gpus = gpus[: n_gpus // 2]
            else:
                faiss_gpus = gpus
                maxsim_gpus = []
        elif n_gpus == 1:
            faiss_gpus = gpus
            maxsim_gpus = []
        else:
            faiss_gpus = []
            maxsim_gpus = []
        return faiss_gpus, maxsim_gpus
