from __future__ import annotations

import logging
import time
from collections import defaultdict
from copy import deepcopy
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rich
import torch
from datasets import Split
from torch import Tensor

from fz_openqa.callbacks.store_results import IDX_COL
from fz_openqa.datamodules.index.dense import FaissIndex
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.datamodules.pipes.predict import OutputFormat
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import cast_to_numpy
from fz_openqa.utils.json_struct import apply_to_json_struct

log = logging.getLogger(__name__)


def log_mem_size(x, key, logger=None):
    if isinstance(x, torch.Tensor):
        mem_size = x.element_size() * x.nelement()
    elif isinstance(x, np.ndarray):
        mem_size = x.dtype.itemsize * x.size
    else:
        raise TypeError(f"Unsupported type {type(x)}")
    mem_size /= 1024 ** 2
    if logger is None:
        rich.print(f"# {key} mem. size={mem_size:.3f} MB")
    else:
        logger.info(f"# {key} mem. size={mem_size:.3f} MB")


class ColbertIndex(FaissIndex):
    _emb2pid: np.ndarray = None
    _use_emb2pid: bool = True

    def _add_batch_to_index(self, batch: Batch, dtype=np.float32):
        """ Add one batch of data to the index """

        # add vector to index
        vector = self._get_vector_from_batch(batch)
        assert isinstance(vector, np.ndarray), f"vector {type(vector)} is not a numpy array"
        assert len(vector.shape) == 3, f"{vector} is not a 3D array"
        self._index.add(vector.reshape(-1, vector.shape[-1]))

        # store token index to original document
        for idx in batch["__idx__"]:
            n_tokens = vector.shape[1]
            pids = np.array(n_tokens * [idx], dtype=np.long)
            if self._emb2pid is None:
                self._emb2pid = pids
            else:
                self._emb2pid = np.concatenate([self._emb2pid, pids])

    def _check_full_index_consistency(self):
        # todo: implement
        pass

        # if not self._index.ntotal * doc_len == self._vectors.num_rows:
        #     raise ValueError(f"The number of vectors in the index ({self._index.ntotal}) is "
        #                      f"not equal to the number of vectors in "
        #                      f"the dataset ({self._vectors.num_rows})")

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
        documents = pq.read_table(str(self.vector_file), memory_map=True)
        documents = self.postprocess(documents[None:None].to_pydict())
        documents = np.array(documents[self.vectors_column_name], dtype=np.float32)
        rich.print(f">> documents:{documents.shape}")
        log_mem_size(documents, "documents")

        # 1. get query vector
        query = self.predict_queries(query, format=OutputFormat.NUMPY, idx=idx, split=split)
        query = self.postprocess(query)
        q_vectors: np.ndarray = query[self.vectors_column_name]
        rich.print(f">> q_vectors:{q_vectors.shape}, k={k}, type={type(q_vectors)}")

        # 2. query the token index
        tok_scores, tok_indices = self._query_to_embedding_ids(q_vectors, max(1, k // 2))

        # 3. get the corresponding document ids
        pids = self._emb2pid[tok_indices]
        pids = self._get_unique_pids(pids)

        # 4. retrieve the vectors for each unique document index
        # the table _vectors (pyarrow) contains the document vectors
        start = time.time()
        d_vectors = self._retrieve_pid_vectors(pids, vectors=documents, dtype=q_vectors.dtype)
        rich.print(f">> vectors:{d_vectors.shape}, retrieval_time={time.time() - start:.3f} s")
        log_mem_size(d_vectors, "vectors")

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
            unique_pid_vectors = self._vectors.take(unique_pids).to_pydict()
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
