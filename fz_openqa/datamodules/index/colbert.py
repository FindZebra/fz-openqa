from __future__ import annotations

import logging
import time
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pyarrow as pa
import rich
import torch
from datasets import Split

from fz_openqa.datamodules.index.dense import FaissIndex
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.datamodules.pipes.predict import OutputFormat
from fz_openqa.utils.datastruct import Batch

log = logging.getLogger(__name__)


def log_mem_size(x, msg, logger=None):
    if isinstance(x, torch.Tensor):
        mem_size = x.element_size() * x.nelement()
    elif isinstance(x, np.ndarray):
        mem_size = x.dtype.itemsize * x.size
    else:
        raise TypeError(f"Unsupported type {type(x)}")
    mem_size /= 1024 ** 2
    if logger is None:
        rich.print(f"{msg} mem. size={mem_size:.3f} MB")
    else:
        logger.info(f"{msg} mem. size={mem_size:.3f} MB")


class ColbertIndex(FaissIndex):
    """
    Implementiation of the Colbert index.

    Notes
    -----
    Assumes the documents to be of the same length!

    """

    _emb2pid: Optional[np.ndarray] = None
    _vectors: Optional[np.ndarray] = None

    @property
    def vectors_table(self) -> pa.Table:
        if self._vectors_table is None:
            self._vectors_table = self._read_vectors_table()
        return self._vectors_table

    @property
    def vectors(self) -> np.ndarray:
        if self._vectors is None:
            vectors_table = self.vectors_table
            self._vectors = self._read_vectors_from_table(vectors_table)
            log_mem_size(self._vectors, "Loaded Colbert vectors", logger=log)
        return self._vectors

    @property
    def emb2pid(self):
        if self._emb2pid is None:
            self._emb2pid = self._build_emb2pid_from_vectors(self.vectors_table)
        return self._emb2pid

    def _iter_vectors(self, vectors_table: pa.Table, max_chunksize=1000) -> np.ndarray:
        for batch in vectors_table.to_batches(max_chunksize=max_chunksize):
            yield self._get_vectors_from_record_batch(batch)

    def _read_vectors_from_table(self, vectors_table: pa.Table, max_chunksize=1000) -> np.ndarray:
        vectors = None
        for vectors_chunk in self._iter_vectors(vectors_table, max_chunksize=max_chunksize):
            if vectors is None:
                vectors = vectors_chunk
            else:
                vectors = np.concatenate([vectors, vectors_chunk], axis=0)
        return vectors

    def _build_emb2pid_from_vectors(self, vectors_table: pa.Table) -> np.ndarray:
        emb2pid = np.arange(vectors_table.num_rows, dtype=np.long)
        one_vec = next(iter(self._iter_vectors(vectors_table, max_chunksize=1)))
        vector_length = one_vec.shape[1]
        emb2pid = np.repeat(emb2pid[:, None], vector_length, axis=1)
        return np.ascontiguousarray(emb2pid.reshape(-1))

    def _get_vectors_from_record_batch(self, batch: pa.RecordBatch) -> np.ndarray:
        batch = batch.to_pydict()
        batch = self.postprocess(batch)
        vec = batch[self.vectors_column_name]
        return np.array(vec, dtype=np.float32)

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
        """

        # 0. load vectors
        if self.in_memory:
            documents = self.vectors
            emb2pid = self.emb2pid
        else:
            documents = self.vectors_table
            emb2pid = self.emb2pid

        # 1. get query vector
        query = self.predict_queries(query, format=OutputFormat.NUMPY, idx=idx, split=split)
        query = self.postprocess(query)
        q_vectors: np.ndarray = query[self.vectors_column_name].astype(np.float32)
        start = time.time()

        # 2. query the token index
        tok_scores, tok_indices = self._query_to_embedding_ids(q_vectors, max(1, k // 2))

        # 3. get the corresponding document ids
        pids = emb2pid[tok_indices]
        pids = self._get_unique_pids(pids)

        # 4. retrieve the vectors for each unique document index
        # the table _vectors (pyarrow) contains the document vectors
        d_vectors = self._retrieve_pid_vectors(pids, vectors=documents, dtype=q_vectors.dtype)

        # 7. apply max sim to the retrieved vectors
        scores = np.einsum("bqh, bkdh -> bkqd", q_vectors, d_vectors)
        # max. over the documents tokens, for each query token
        scores = np.amax(scores, axis=-1)
        # avg over all query tokens
        scores = scores.mean(axis=-1)
        # set the score to -inf for the negative pids
        scores[pids < 0] = -np.inf

        # 8. take the top-k results given the MaxSim score
        maxsim_idx = np.argsort(-scores, axis=-1)[:, :k]

        # 9. fetch the corresponding document indices and return
        maxsim_scores = np.take_along_axis(scores, maxsim_idx, axis=1)
        maxsim_pids = np.take_along_axis(pids, maxsim_idx, axis=1)

        log.info(
            f"Colbert: "
            f"retrieval_time={time.time() - start:.3f}s, "
            f"k={k}, d_vectors={d_vectors.shape}"
        )
        return SearchResult(
            score=maxsim_scores,
            index=maxsim_pids,
            dataset_size=self.dataset_size,
            k=k,
        )

    def _get_unique_pids(self, pids, fill_value=-1):
        # pids = np.unique(pids, axis=1)
        upids = [np.unique(r) for r in pids]
        max_length = max(len(r) for r in upids)

        def _pad(r):
            return np.pad(r, (0, max_length - len(r)), mode="constant", constant_values=fill_value)

        return np.concatenate([_pad(p)[None] for p in upids])

    def _retrieve_pid_vectors(
        self, pids: np.ndarray, *, vectors: pa.Table | np.ndarray, dtype: np.dtype
    ) -> np.ndarray:
        """Retrieve the vectors for the given pids."""
        if isinstance(vectors, pa.Table):
            batch_size, depth = pids.shape

            # flatten and get unique documents
            pids = pids.reshape(-1)
            unique_pids, indices = np.unique(pids, return_inverse=True)
            unique_pids[unique_pids < 0] = 0

            # retrieve unique document vectors and cast as array
            unique_pid_vectors = self._vectors_table.take(unique_pids).to_pydict()
            unique_pid_vectors = self.postprocess(unique_pid_vectors)[self.vectors_column_name]
            unique_pid_vectors = np.array(unique_pid_vectors, dtype=dtype)

            # reshape and return
            pid_vectors = unique_pid_vectors[indices]
            return pid_vectors.reshape(
                (
                    batch_size,
                    depth,
                )
                + pid_vectors.shape[1:]
            )
        elif isinstance(vectors, np.ndarray):
            return vectors[pids]
        else:
            raise TypeError(f"vectors type {type(vectors)} not supported")

    def _query_to_embedding_ids(
        self, Q: np.ndarray, faiss_depth: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query the faiss index for each embedding vector"""
        num_queries, embeddings_per_query, dim = Q.shape
        Q = Q.reshape(-1, Q.shape[-1])
        scores, embedding_ids = self._index.search(Q, faiss_depth)

        scores = scores.reshape(num_queries, -1)
        embedding_ids = embedding_ids.reshape(num_queries, -1)
        return scores, embedding_ids
