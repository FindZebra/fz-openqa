from __future__ import annotations

import logging
from typing import List
from typing import Optional

import faiss.contrib.torch_utils  # type: ignore
import rich
import torch
from datasets import Split
from torch import Tensor

from fz_openqa.datamodules.index.dense import DenseIndex
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.datamodules.index.utils.io import build_emb2pid_from_vectors
from fz_openqa.datamodules.index.utils.io import log_mem_size
from fz_openqa.datamodules.index.utils.io import read_vectors_from_table
from fz_openqa.datamodules.index.utils.maxsim.maxsim import MaxSim
from fz_openqa.datamodules.pipes import Predict
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import OutputFormat
from fz_openqa.utils.pretty import pprint_batch
from fz_openqa.utils.tensor_arrow import TensorArrowTable

# required to allow searching faiss with tensors

log = logging.getLogger(__name__)


class ColbertIndex(DenseIndex):
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
    _max_sim: Optional[MaxSim] = None
    _is_gpu: bool = False
    _max_add_per_gpu = 1 << 25
    _max_num_proc: int = 1

    no_fingerprint: List[str] = DenseIndex.no_fingerprint + [
        "_max_sim",
        "_vectors" "_emb2pid",
        "in_memory",
        "_is_gpu",
        "_max_add_per_gpu",
        "maxsim_chunksize",
    ]

    def __init__(
        self,
        *args,
        p: int = 100,
        maxsim_chunksize: int = 10000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.p = p
        self.maxsim_chunksize = maxsim_chunksize

    def _call_batch(
        self,
        query: Batch,
        idx: Optional[List[int]] = None,
        k: Optional[int] = None,
        output_format: Optional[OutputFormat] = None,
        **kwargs,
    ) -> Batch:
        """
        Search the index for a batch of examples (query).

        Filter the incoming batch using the same pipe as the one
        used to build the index."""
        k = k or self.k

        # load MaxSim
        if self._max_sim is None:
            self._init_maxsim()

        # pprint_batch(query, f"{type(self)} : query")

        doc_ids = query.get("question.document_idx", None)
        query = self._preprocess_query(query, idx=idx, **kwargs)
        q_vectors: Tensor = query[Predict.output_key]

        # search the index by chunk
        batch_size = len(q_vectors)
        if self.max_chunksize is not None:
            eff_batch_size = min(max(1, self.max_chunksize), batch_size)
        else:
            eff_batch_size = batch_size

        # send all queries to MaxSimParallel
        search_results = None
        for idx, i in enumerate(range(0, batch_size, eff_batch_size)):
            chunk_i = q_vectors[i : i + eff_batch_size]
            if doc_ids is not None:
                doc_ids_i = doc_ids[i : i + eff_batch_size]
            else:
                doc_ids_i = None
            data = self._max_sim(chunk_i, k=k, p=self.p, idx=idx, doc_ids=doc_ids_i)

            # cast to SearchResult
            r = SearchResult(
                score=data.scores.cpu().numpy(),
                index=data.pids.cpu().numpy(),
                dataset_size=self.dataset_size,
                k=k,
            )
            if search_results is None:
                search_results = r
            else:
                search_results += r

        return self._format_output(search_results, output_format=output_format)

    def _init_maxsim(self):
        faiss_devices, maxsim_devices = self._allocate_gpus(faiss.get_num_gpus())
        self._max_sim = MaxSim(
            token_index=self.index_path,
            vectors=self.vectors_table,
            emb2pid=self.emb2pid,
            ranking_devices=maxsim_devices,
            faiss_devices=faiss_devices,
            max_chunksize=self.maxsim_chunksize,
        )
        self._max_sim.cuda()

    def free_memory(self):
        super(ColbertIndex, self).free_memory()
        if hasattr(self, "_max_sim") and self._max_sim is not None:
            del self._max_sim
            self._max_sim = None

    def _train_ends(self):
        self.free_memory()

    def __del__(self):
        self.free_memory()
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
            self._vectors = read_vectors_from_table(vectors_table)
            log_mem_size(self._vectors, "Loaded Colbert vectors", logger=log)
        return self._vectors

    @property
    def emb2pid(self) -> Tensor:
        if self._emb2pid is None:
            self._emb2pid = build_emb2pid_from_vectors(self.vectors_table)
        return self._emb2pid

    def _allocate_gpus(self, n_gpus):
        """Allocate GPUs to the faiss index and to max_sim"""
        gpus = list(range(n_gpus))
        if self.keep_faiss_on_cpu:
            faiss_gpus = []
            maxsim_gpus = gpus
        elif n_gpus > 1:
            if not self.keep_faiss_on_cpu:
                n_maxsim = min(-(-n_gpus // 2), n_gpus - 1)
                faiss_gpus = gpus[n_maxsim:]
                maxsim_gpus = gpus[:n_maxsim]
            else:
                faiss_gpus = []
                maxsim_gpus = gpus
        elif n_gpus == 1:
            faiss_gpus = gpus
            maxsim_gpus = gpus
        else:
            faiss_gpus = []
            maxsim_gpus = []
        return faiss_gpus, maxsim_gpus
