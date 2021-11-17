import logging
from functools import singledispatchmethod
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Union

import faiss
import numpy as np
import pytorch_lightning as pl
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split
from faiss.swigfaiss import Index as FaissSwigIndex
from pytorch_lightning import Trainer
from rich.progress import track
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

from fz_openqa.callbacks.store_results import IDX_COL
from fz_openqa.datamodules.index.base import Index
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import Predict
from fz_openqa.datamodules.pipes import RenameKeys
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.utils.functional import infer_batch_size
from fz_openqa.utils.functional import is_index_contiguous

logger = logging.getLogger(__name__)


class FaissIndex(Index):
    """Dense indexing using faiss"""

    # todo: remove need to store the `document.row_idx`

    vectors_column_name = "__vectors__"
    _dtype: np.dtype = np.float32
    _index: FaissSwigIndex = None
    model: Callable = None

    def __init__(
        self,
        dataset: Dataset,
        *,
        model: pl.LightningModule,
        trainer: Optional[Trainer] = None,
        nlist: int = 10,
        index_key: str = "document.row_idx",
        model_output_keys: List[str],
        loader_kwargs: Optional[Dict] = None,
        collate_pipe: Pipe = None,
        metric_type: str = faiss.METRIC_INNER_PRODUCT,
        persist_cache: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):

        #  save fingerprints
        self._model_fingerprint = get_fingerprint(model)
        self._dataset_fingerprint = dataset._fingerprint

        # model and params
        self.model = model
        self.nlist = nlist
        self.metric_type = metric_type

        self.index_key = index_key
        self.model_output_keys = model_output_keys

        # trainer and dataloader
        self.trainer = trainer
        self.loader_kwargs = loader_kwargs

        # collate pipe use to convert dataset rows into a batch
        self.collate = collate_pipe or Collate()

        # process pipe, used to process a batch with the model
        # warning: this is actually not used when using the trainer
        self.persist_cache = persist_cache
        self.cache_dir = cache_dir
        self.predict_docs = Predict(model=self.model)
        self.predict_queries = Predict(model=self.model)

        # postprocessing: rename the model outputs `model_output_keys` to `vectors_column_name`
        self.postprocess = self.get_rename_output_names_pipe(
            inputs=self.model_output_keys, output=self.vectors_column_name
        )

        # call the super: build the index
        super(FaissIndex, self).__init__(dataset=dataset, **kwargs)

    @property
    def is_indexed(self):
        return self._index is None or self._index.is_trained

    def build(self, dataset: Dataset, **kwargs):
        """Index a dataset."""

        # if a trainer is available, use it to process and cache the whole dataset
        loader_args = {}
        if self.trainer is not None:
            self._predict_and_cache(dataset, predict=self.predict_docs, collate_fn=self.collate)
            loader_args["batch_size"] = 1000

        # instantiate the base data loader
        loader = self._init_loader(dataset, **loader_args)

        # process batches: this returns a generator, so batches will be processed
        # as they are consumed in the following loop
        desc = "Ingest Faiss index"
        if self.trainer is not None:
            desc += " (loading vectors from cache)"
        processed_batches = self._process_batches(loader, predict=self.predict_docs, desc=desc)

        # build an iterator over the processed batches
        it = iter(processed_batches)

        # init the faiss index and add the 1st batch
        batch = next(it)
        self._init_index(batch)
        self._add_batch_to_index(batch)

        # iterate through the remaining batches and add them to the index
        while True:
            try:
                batch = next(it)
                self._add_batch_to_index(batch)
            except Exception:
                break

        logger.info(
            f"Index is_trained={self._index.is_trained}, "
            f"size={self._index.ntotal}, type={type(self._index)}"
        )

    def _predict_and_cache(
        self, dataset: Union[Dataset, DatasetDict], *, predict: Predict, collate_fn: Callable
    ):
        predict.invalidate_cache()
        predict.cache(
            dataset,
            trainer=self.trainer,
            collate_fn=collate_fn,
            loader_kwargs=self.loader_kwargs,
            persist=self.persist_cache,
            cache_dir=self.cache_dir,
        )

    def _init_index(self, batch):
        """
        Initialize the Index
        # todo: @idariis : You can modify this method to add more advanced indexes (e.g. IVF)
        """
        vectors = batch[self.vectors_column_name]
        assert len(vectors.shape) == 2
        self._index = faiss.IndexFlat(vectors.shape[-1], self.metric_type)

    def _add_batch_to_index(self, batch: Batch):
        """
        Add one batch of data to the index
        """
        # check the ordering of the indexes
        indexes = batch[IDX_COL]
        msg = f"Indexes are not contiguous (i.e. 1, 2, 3, 4),\nindexes={indexes}"
        assert is_index_contiguous(indexes), msg
        msg = (
            f"The stored index and the indexes are not contiguous, "
            f"\nindex_size={self._index.ntotal}, first_index={indexes[0]}"
        )
        assert self._index.ntotal == indexes[0], msg

        # add the vectors to the index
        vector = self._get_vector_from_batch(batch)
        assert isinstance(vector, np.ndarray), f"vector {type(vector)} is not a numpy array"
        assert len(vector.shape) == 2, f"{vector} is not a 2D array"
        self._index.add(vector)

    def _query_index(self, query: Batch, *, k: int) -> SearchResult:
        """Query the index given a batch of data"""
        vector = self._get_vector_from_batch(query)
        score, indices = self._index.search(vector, k)
        return SearchResult(score=score, index=indices, dataset_size=self.dataset_size)

    @singledispatchmethod
    def search(
        self,
        query: Batch,
        *,
        idx: Optional[List[int]] = None,
        split: Optional[Split] = None,
        k: int = 1,
        **kwargs,
    ) -> SearchResult:
        """Search the index using the `query` and
        return the index of the results within the original dataset."""
        query = self.predict_queries(query, idx=idx, split=split)
        query = self.postprocess(query)
        return self._query_index(query, k=k)

    @search.register(Dataset)
    def _(
        self, query: Dataset, *, k: int = 1, collate_fn: Callable, **kwargs
    ) -> Iterator[SearchResult]:
        """This method is called if `query` is of type Dataset"""
        if self.trainer:
            self.cache_query_dataset(query, collate_fn=collate_fn)

        loader = self._init_loader(query)
        return self.search(loader, k=k, **kwargs)

    @search.register(DatasetDict)
    def _(self, query: DatasetDict, *, k: int = 1, **kwargs) -> Dict[Split, Iterator[SearchResult]]:
        """This method is called if `query` is of type DatasetDict"""
        return {split: self.search(dset, k=k, **kwargs) for split, dset in query.items()}

    def cache_query_dataset(
        self, dataset: Union[Dataset, DatasetDict], *, collate_fn: Callable, **kwargs
    ):
        self._predict_and_cache(dataset, predict=self.predict_queries, collate_fn=collate_fn)

    def _get_vector_from_batch(self, batch: Batch) -> np.ndarray:
        """Get and cast the vector from the batch"""
        vector: np.ndarray = batch[self.vectors_column_name]
        vector = vector.astype(self._dtype)
        return vector

    def _init_loader(self, dataset, **kwargs):
        _kwargs = self.loader_kwargs.copy()
        for k, v in kwargs.items():
            _kwargs[k] = v
        loader = DataLoader(
            dataset,
            collate_fn=self.collate,
            shuffle=False,
            **_kwargs,
        )
        return loader

    def _process_batches(
        self,
        loader: DataLoader,
        *,
        predict: Predict,
        desc: str = "Processing dataset",
        progress_bar: bool = True,
    ) -> Iterable[Batch]:

        """
        This method iterates over batches, which for each batch:
        1. process the batch using the model (or load from cache)
        2. postprocess the output of the model (renaming keys, filtering keys, etc)

        This function processes batches as they are loaded from the dataloader.
        """

        # modify the dataloader to return __idx__ aside from the original data
        assert isinstance(
            loader.sampler, SequentialSampler
        ), "Cannot handle DataLoader with shuffle=True."

        if progress_bar:
            loader = track(iter(loader), description=desc)

        # process each batch sequentially using `self.process`
        i = 0
        for batch in loader:
            bs = infer_batch_size(batch)
            batch = predict(batch, idx=list(range(i, i + bs)))
            batch = self.postprocess(batch)
            i += bs
            yield batch

    def get_rename_output_names_pipe(self, inputs: List[str], output: str) -> Pipe:
        """Format the output of the model"""
        return Sequential(
            RenameKeys({key: output for key in inputs}),
            FilterKeys(In([output, IDX_COL])),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # state["model"] = None # todo: check this
        state["trainer"] = None
        state["_index"] = faiss.serialize_index(state["_index"])
        return state

    def __setstate__(self, state):
        state["_index"] = faiss.deserialize_index(state["_index"])
        self.__dict__.update(state)
