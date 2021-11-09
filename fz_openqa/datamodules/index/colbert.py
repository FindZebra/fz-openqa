import logging
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import dill
import faiss
import numpy as np
from datasets import Dataset
from faiss.swigfaiss import Index as FaissSwigIndex
from pytorch_lightning import Trainer
from rich.progress import track
from torch.nn import Module
from torch.utils.data import DataLoader

from fz_openqa.datamodules.index.base import Index
from fz_openqa.datamodules.index.base import SearchResult
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import Forward
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import RenameKeys
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import ToNumpy
from fz_openqa.datamodules.pipes.control.filter_keys import KeyIn
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.fingerprint import get_fingerprint

logger = logging.getLogger(__name__)


class ColbertIndex(Index):
    """Dense indexing using faiss"""

    vectors_column_name = "__vectors__"
    _index: FaissSwigIndex = None
    model: Callable = None

    def __init__(
        self,
        dataset: Dataset,
        *,
        model: Callable,
        trainer: Optional[Trainer] = None,
        nlist: int = 10,
        index_key: str = "document.row_idx",
        model_output_keys: List[str],
        batch_size: int = 10,
        collate_pipe: Pipe = Collate(),
        metric_type: str = faiss.METRIC_INNER_PRODUCT,
        **kwargs,
    ):

        #  save fingerprints
        self._model_fingerprint = get_fingerprint(model)
        self._datadset_fingerprint = dataset._fingerprint

        # model and params
        self.model = model
        self.nlist = nlist
        self.metric_type = metric_type
        self.batch_size = batch_size
        self.index_key = index_key
        self.model_output_keys = model_output_keys

        # Pipes
        self.collate = collate_pipe
        self.process = self.get_batch_processing_pipe(model)

        super(ColbertIndex, self).__init__(dataset=dataset, **kwargs)

    def build(self, dataset: Dataset, **kwargs):
        """Index a dataset."""

        # todo: refactor to apply the preprocessing pipe on the fly.
        #  So caching the new dataset can be avoided.
        # todo: get rid of dataset, use a dataloader / check compatibility
        #  with lightning: potentially solves all the issues
        #  with slow loading of the batches on device!
        # todo: safe pickle and multiprocessing (save index to file)
        # cols_to_drop = list(sorted(set(dataset.column_names) - {self.index_key, self.text_key}))
        # dataset = dataset.remove_columns(cols_to_drop)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=1,
            collate_fn=self.collate,
            shuffle=False,
        )
        it = track(iter(loader), description="Indexing dataset")

        # init faiss index
        batch = next(it)
        batch = self.process(batch)
        self._init_index(batch)
        self._add_batch_to_index(batch)

        # iterate through batches
        while True:
            try:
                batch = next(it)
                batch = self.process(batch)
                self._add_batch_to_index(batch)
            except Exception:
                break

        logger.info(
            f"Index is_trained={self._index.is_trained}, "
            f"size={self._index.ntotal}, type={type(self._index)}"
        )

    def dill_inspect(self) -> Dict[str, bool]:
        """check if the module can be pickled."""
        return {
            "__all__": dill.pickles(self),
            **{k: dill.pickles(v) for k, v in self.__getstate__().items()},
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_index"] = faiss.serialize_index(state["_index"])
        return state

    def __setstate__(self, state):
        state["_index"] = faiss.deserialize_index(state["_index"])
        self.__dict__.update(state)

    def _init_index(self, batch):
        vectors = batch[self.vectors_column_name]
        assert len(vectors.shape) == 2
        self._index = faiss.IndexFlat(vectors.shape[-1], self.metric_type)

    def _add_batch_to_index(self, batch: Batch):
        vectors = batch[self.vectors_column_name]
        self._index.add(np.ascontiguousarray(vectors))

    @property
    def is_indexed(self):
        return self._index is None or self._index.is_trained

    def search(
        self,
        query: Batch,
        *,
        k: int = 1,
        **kwargs,
    ) -> SearchResult:
        """Search the index using the `query` and
        return the index of the results within the original dataset."""
        query = self.process(query)
        vectors = query[self.vectors_column_name]
        vectors = np.ascontiguousarray(vectors)
        score, indices = self._index.search(vectors, k)
        return SearchResult(score=score, index=indices)

    def get_batch_processing_pipe(self, model: Union[Callable, Module]) -> Pipe:
        """returns a pipe that allows processing a batch of data using the model."""
        return Sequential(
            Forward(model=model, output_key=self.vectors_column_name),
            RenameKeys({key: self.vectors_column_name for key in self.model_output_keys}),
            FilterKeys(KeyIn([self.vectors_column_name])),
            ToNumpy(),
        )
