import io
import logging
from typing import Callable
from typing import List
from typing import Union

import faiss
import numpy as np
import rich
import torch
import xxhash
from datasets import Dataset
from faiss.swigfaiss import Index as FaissSwigIndex
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
from fz_openqa.utils.pretty import pprint_batch

logger = logging.getLogger(__name__)


def fingerprint_model(model):
    hasher = xxhash.xxh64()
    state = model.state_dict()
    for (k, v) in sorted(state.items(), key=lambda x: x[0]):
        # it did not work without hashing the tensor
        hasher.update(k)
        buff = io.BytesIO()
        torch.save(v, buff)
        buff.seek(0)
        hasher.update(buff.read())

    return hasher.hexdigest()


class FaissIndex(Index):
    """Dense indexing using faiss"""

    vectors_column_name = "__vectors__"
    _index: FaissSwigIndex = None
    model: Callable = None

    def __init__(
        self,
        dataset: Dataset,
        *,
        model: Callable,
        nlist: int = 10,
        index_key: str = "document.row_idx",
        model_output_keys: List[str],
        batch_size: int = 10,
        collate_pipe: Pipe = Collate(),
        metric_type: str = faiss.METRIC_INNER_PRODUCT,
        **kwargs,
    ):
        self.model = model
        self._model_fingerprint = fingerprint_model(model)
        self.nlist = nlist
        self.batch_size = batch_size
        self.index_key = index_key
        self.model_output_keys = model_output_keys
        self.collate = collate_pipe
        self.process = self.get_batch_processing_pipe(model)
        self.metric_type = metric_type

        super(FaissIndex, self).__init__(dataset=dataset, **kwargs)

    def build(self, dataset: Dataset, **kwargs):
        """Index a dataset."""

        # todo: refactor to apply the preprocessing pipe on the fly.
        #  So caching the new dataset can be avoided.
        # get rid of dataset, use a dataloader / check compatibility
        # with lightning: potentially solves all the issues
        # with slow loading of the batches on device!
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
        vectors = batch[self.vectors_column_name]
        assert len(vectors.shape) == 2
        self._index = faiss.IndexFlat(vectors.shape[-1], self.metric_type)
        self._index.add(np.ascontiguousarray(vectors))

        # iterate through batches
        while True:
            try:
                batch = next(it)
                batch = self.process(batch)
                vectors = batch[self.vectors_column_name]
                self._index.add(np.ascontiguousarray(vectors))
            except Exception:
                break

        logger.info(
            f"Index is_trained={self._index.is_trained}, "
            f"size={self._index.ntotal}, type={type(self._index)}"
        )

    @property
    def is_indexed(self):
        return self._index is None or self._index.is_trained

    # def __getstate__(self):

    def search(
        self,
        query: Batch,
        *,
        k: int = 1,
        model: Union[Callable, Module] = None,
        **kwargs,
    ) -> SearchResult:
        """Search the index using the `query` and
        return the index of the results within the original dataset."""
        assert self.dataset is not None
        model = model or self.model
        assert model is not None

        query = self.get_batch_processing_pipe(model=model)(query)

        results = self.dataset.get_nearest_examples_batch(
            self.vectors_column_name, query[self.vectors_column_name], k=k
        )

        return SearchResult(
            score=results.total_scores, index=[r[self.index_key] for r in results.total_examples]
        )

    def get_batch_processing_pipe(self, model: Union[Callable, Module]) -> Pipe:
        """returns a pipe that allows processing a batch of data using the model."""
        return Sequential(
            Forward(model=model, output_key=self.vectors_column_name),
            RenameKeys({key: self.vectors_column_name for key in self.model_output_keys}),
            FilterKeys(KeyIn([self.vectors_column_name])),
            ToNumpy(),
        )
