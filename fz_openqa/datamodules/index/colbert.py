import logging
from typing import List

import faiss
import numpy as np
import pytorch_lightning
import rich
from datasets import Dataset
from faiss.swigfaiss import Index as FaissSwigIndex
from pytorch_lightning import Trainer
from rich import status
from rich.progress import track
from torch.utils import data
from torch.utils.data import DataLoader

from fz_openqa.callbacks.store_results import StorePredictionsCallback
from fz_openqa.datamodules.index.base import Index
from fz_openqa.datamodules.index.dense import FaissIndex
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import pprint_batch

log = logging.getLogger(__name__)

DEF_LOADER_KWARGS = {"batch_size": 10, "num_workers": 2, "pin_memory": True}
DEFAULT_FAISS_KWARGS = {
    "metric_type": faiss.METRIC_L2,
    "n_list": 32,
    "m": 16,
    "n_bits": 8,
}


class ColbertIndex(FaissIndex):
    def __init__(self, dataset: Dataset, **kwargs):
        """
        Reused init from parent class (FaissIndex)
        """
        # call the super: build the index
        super(ColbertIndex, self).__init__(dataset=dataset, **kwargs)

    def _init_index(self, batch):
        """
        Initialize the index and train on first batch of data

        Parameters
        ----------
        vectors
            The dataset to index.
        metric_type
            Distance metric, most common is METRIC.L2
        n_list
            The number of cells (partitions). Typical values is sqrt(N)
        m
            The number of sub-vectors. Typically this is 8, 16, 32, etc.
        n_bits
            Bits per subvector. Typically 8, so that each sub-vector is encoded by 1 byte
        dim
            The dimension of the input vectors

        """
        vectors: np.ndarray = batch[self.vectors_column_name]
        vectors = vectors.astype(np.float32)
        assert len(vectors.shape) == 2
        metric_type = self.faiss_args["metric_type"]
        n_list = self.faiss_args["n_list"]
        m = self.faiss_args["m"]
        n_bits = self.faiss_args["n_bits"]
        self.dim = vectors.shape[-1]
        assert self.dim % m == 0, "m must be a divisor of dim"

        quantizer = faiss.IndexFlatL2(self.dim)
        self._index = faiss.IndexIVFPQ(quantizer, self.dim, n_list, m, n_bits, metric_type)
        self._index.train(vectors)
        assert self._index.is_trained is True, "Index is not trained"
        log.info(f"Index is trained: {self._index.is_trained}")

    def _add_batch_to_index(self, batch: Batch, dtype=np.float32):
        """ Add one batch of data to the index """
        # check indexes
        # self._check_index_consistency(idx)

        # add vector to index
        rich.print("Printing batch...")
        pprint_batch(batch)
        vector = self._get_vector_from_batch(batch)
        assert isinstance(vector, np.ndarray), f"vector {type(vector)} is not a numpy array"
        assert len(vector.shape) == 2, f"{vector} is not a 2D array"
        self._index.add(vector)

        # store token index to original document
        self.tok2doc = []
        for idx in batch["__idx__"]:
            ids = np.linspace(idx, idx, num=self.dim, dtype="int32").tolist()
            self.tok2doc.extend(ids)
        rich.print(f"[red]Total number of indices: {self._index.ntotal}")

    def search(
        self,
        query: Batch,
        *,
        k: int = 1,
        **kwargs,
    ) -> SearchResult:
        """
        Search the index using the `query` and
        return the index of the results within the original dataset.

        Note
        ----
        Scores are distances, i.e. the smaller the better
        """
        rich.print(self.tok2doc)
        query = self.predict_queries(query)
        query = self.postprocess(query)
        vectors = query[self.vectors_column_name]
        scores, indices = self._index.search(vectors, k)

        doc_indices = set(self.tok2doc[index] for index in indices.flatten("C"))

        # todo: apply MaxSim to filter related documents further

        return SearchResult(score=scores, index=doc_indices)
