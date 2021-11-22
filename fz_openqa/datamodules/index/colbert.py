import logging
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

import dill
import faiss
import numpy as np
import pytorch_lightning
import rich
from datasets import Dataset
from faiss.swigfaiss import Index as FaissSwigIndex
from pytorch_lightning import Trainer
from rich.progress import track
from torch.utils import data
from torch.utils.data import DataLoader

from fz_openqa.callbacks.store_results import StorePredictionsCallback
from fz_openqa.datamodules.index.base import Index
from fz_openqa.datamodules.index.dense import FaissIndex
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import Forward
from fz_openqa.datamodules.pipes import Parallel
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import RenameKeys
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import ToNumpy
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.fingerprint import get_fingerprint

logger = logging.getLogger(__name__)

DEF_LOADER_KWARGS = {"batch_size": 10, "num_workers": 2, "pin_memory": True}


class ColbertIndex(FaissIndex):
    def __init__(self, dataset: Dataset, partitions: int, **kwargs):
        """
        todo: @idariis : this needs to be adapted for Colbert
        the init should mostly reuse the one from the parent class (FaissIndex)
        you might need to init the token index (to store the index of the original document)
        """
        # initializing parameters for faiss index
        # self.dim = dim
        self.partitions = partitions

        # call the super: build the index
        super(ColbertIndex, self).__init__(dataset=dataset, **kwargs)

    def _init_index(self, batch):
        """
        Initialize the index
         # todo: @idariis : this needs to be adapted for Colbert
        """
        vectors = batch[self.vectors_column_name]
        assert len(vectors.shape) == 2
        dim = vectors.shape[-1]
        quantizer = faiss.IndexFlatL2(dim)
        self._index = faiss.IndexIVFPQ(quantizer, dim, self.partitions, 2, 2, self.metric_type)

    def train(self, vector, dtype=np.float32):
        """
        Train index on data
        """
        if self._index.is_trained is False:
            print("#> Training index")
            self._index.train(vector)
            logger.info(
                f"Index is_trained={self._index.is_trained}, "
                f"size={self._index.ntotal}, type={type(self._index)}"
            )
        else:
            return print(f"Index is trained={self._index.is_trained}")

    def _add_batch_to_index(self, batch: Batch, dtype=np.float32):
        """
        Add one batch of data to the index
         # todo: @idariis : this needs to be adapted for Colbert
        """
        # check indexes
        indexes = batch[self.index_key]
        msg = f"Indexes are not contiguous (i.e. 1, 2, 3, 4),\nindexes={indexes}"
        assert all(indexes[:-1] + 1 == indexes[1:]), msg
        msg = (
            f"The stored index and the indexes are not contiguous, "
            f"\nindex_size={self._index.ntotal}, first_index={indexes[0]}"
        )
        assert self._index.ntotal == indexes[0], msg

        # add the vectors
        vector: np.ndarray = batch[self.vectors_column_name]
        assert isinstance(vector, np.ndarray), f"vector {type(vector)} is not a numpy array"
        assert len(vector.shape) == 2, f"{vector} is not a 2D array"
        vector = vector.astype(dtype)
        self.train(vector=vector)
        if self._index.is_trained is True:
            self._index.add(vector)

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
        # todo: @idariis : this needs to be adapted for Colbert
        """
        query = self.process(query)
        query = self.postprocess(query)
        vectors = query[self.vectors_column_name]
        score, indices = self._index.search(vectors, k)
        return SearchResult(score=score, index=indices)
