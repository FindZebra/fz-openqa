import logging

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

logger = logging.getLogger(__name__)

DEF_LOADER_KWARGS = {"batch_size": 10, "num_workers": 2, "pin_memory": True}
DEFAULT_FAISS_KWARGS = {
    "metric_type": faiss.METRIC_INNER_PRODUCT,
    "n_list": 32,
    "m": 16,
    "n_bits": 8,
}


class ColbertIndex(FaissIndex):
    def __init__(self, dataset: Dataset, **kwargs):
        """
        todo: @idariis : this needs to be adapted for Colbert
        the init should mostly reuse the one from the parent class (FaissIndex)
        you might need to init the token index (to store the index of the original document)
        """

        # call the super: build the index
        super(ColbertIndex, self).__init__(dataset=dataset, **kwargs)

    def _init_index(self, batch):
        """
        Initialize the index
         # todo: @idariis : this needs to be adapted for Colbert
        """
        vectors = batch[self.vectors_column_name]
        assert len(vectors.shape) == 2
        metric_type = self.faiss_args["metric_type"]
        n_list = self.faiss_args["n_list"]
        m = self.faiss_args["m"]
        n_bits = self.faiss_args["n_bits"]
        dim = vectors.shape[-1]
        assert dim % m == 0, "m must be a divisor of dim"

        quantizer = faiss.IndexFlatL2(dim)
        self._index = faiss.IndexIVFPQ(quantizer, dim, n_list, m, n_bits, metric_type)

    def _train(self, vector):
        """
        Train faiss index on data

        Parameters
        ----------
        vector
            vector from batch

        """
        self._index.train(vector)
        assert self._index.is_trained is True, "Index is not trained"
        print("Done training!")

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

        # train index and add vector to index
        vector: np.ndarray = batch[self.vectors_column_name]
        assert isinstance(vector, np.ndarray), f"vector {type(vector)} is not a numpy array"
        assert len(vector.shape) == 2, f"{vector} is not a 2D array"
        vector = vector.astype(dtype)
        self._train(vector)
        if self._index.is_trained is True:
            self._index.add(vector)
            # store token index to original document
            self.tok2doc = []
            for idx in vector["document.row_idx"]:
                ids = np.linspace(idx, idx, num=198, dtype="int32").tolist()
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
        # todo: @idariis : this needs to be adapted for Colbert
        """
        query = self.process(query)
        query = self.postprocess(query)
        vectors = query[self.vectors_column_name]
        scores, indices = self._index.search(vectors, k)

        indices_flat = [i for sublist in indices for i in sublist]
        doc_indices = [self.tok2doc[index] for index in indices_flat]
        doc_indices = set(doc_indices)

        return SearchResult(score=scores, index=doc_indices)
