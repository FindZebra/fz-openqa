import logging
from functools import singledispatchmethod
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sized

import faiss
import numpy as np
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split
from faiss.swigfaiss import Index as FaissSwigIndex
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from rich.progress import track
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import SequentialSampler

from fz_openqa.callbacks.store_results import IDX_COL
from fz_openqa.callbacks.store_results import StorePredictionsCallback
from fz_openqa.datamodules.index.base import Index
from fz_openqa.datamodules.index.base import SearchResult
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
from fz_openqa.utils.functional import is_index_contiguous

logger = logging.getLogger(__name__)

DEFAULT_LOADER_KWARGS = {"batch_size": 10, "num_workers": 2, "pin_memory": True}


class AddRowIdx(TorchDataset):
    """This class is used to add the column `IDX_COL` to the batch"""

    def __init__(self, dataset: Sized):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item) -> Batch:
        batch = self.dataset[item]
        batch[IDX_COL] = item
        return batch


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
        model: Callable,
        trainer: Optional[Trainer] = None,
        nlist: int = 10,
        index_key: str = "document.row_idx",
        model_output_keys: List[str],
        loader_kwargs: Optional[Dict] = None,
        collate_pipe: Pipe = None,
        metric_type: str = faiss.METRIC_INNER_PRODUCT,
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
        self.loader_kwargs = loader_kwargs or DEFAULT_LOADER_KWARGS

        # collate pipe use to convert dataset rows into a batch
        self.collate = collate_pipe or Collate()

        # process pipe, used to process a batch with the model
        # warning: this is actually not used when using the trainer
        self.process = Sequential(
            Parallel(Forward(model=model), FilterKeys(In([IDX_COL]))), ToNumpy()
        )

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

        # instantiate the base data loader
        loader = self._init_loader(dataset)

        # process batches: this returns a generator, so batches will be processed
        # as they are consumed in the following loop
        processed_batches = self._process_batches(loader, trainer=self.trainer, key="document")

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
        return SearchResult(score=score, index=indices)

    @singledispatchmethod
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
        query = self.postprocess(query)
        return self._query_index(query, k=k)

    @search.register(DataLoader)
    def _(self, query: DataLoader, *, k: int = 1, **kwargs) -> Iterator[SearchResult]:
        """This method is called if `query` is of type DataLoader"""
        processed_batches = self._process_batches(query, trainer=self.trainer, key="question")
        for batch in processed_batches:
            yield self._query_index(batch, k=k)

    @search.register(Dataset)
    def _(self, query: Dataset, *, k: int = 1, **kwargs) -> Iterator[SearchResult]:
        """This method is called if `query` is of type Dataset"""
        loader = self._init_loader(query)
        return self.search(loader, k=k, **kwargs)

    @search.register(DatasetDict)
    def _(self, query: DatasetDict, *, k: int = 1, **kwargs) -> Dict[Split, Iterator[SearchResult]]:
        """This method is called if `query` is of type DatasetDict"""
        return {split: self.search(dset, k=k, **kwargs) for split, dset in query.items()}

    def _get_vector_from_batch(self, batch: Batch) -> np.ndarray:
        """Get and cast the vector from the batch"""
        vector: np.ndarray = batch[self.vectors_column_name]
        vector = vector.astype(self._dtype)
        return vector

    def _init_loader(self, dataset):
        loader = DataLoader(
            self._wrap_dataset(dataset),
            collate_fn=self._wrap_collate_fn(self.collate),
            shuffle=False,
            **self.loader_kwargs,
        )
        return loader

    def _process_batches(
        self,
        loader: DataLoader,
        trainer: Optional[Trainer] = None,
        progress_bar: bool = True,
        key: str = "document",
    ) -> Iterable[Batch]:

        """
        This method iterates over batches, which for each batch:
        1. process the batch using the model
        2. postprocess the output of the model (renaming keys, filtering keys, etc)

        This function processes batches as they are loaded from the dataloader.
        """

        # modify the dataloader to return __idx__ aside from the original data
        loader = self._wrap_dataloader(loader)
        assert isinstance(
            loader.sampler, SequentialSampler
        ), "Cannot handle DataLoader with shuffle=True."

        if trainer is not None and isinstance(self.model, LightningModule):
            # retrieve the callback used to store the results
            store_results_callback = self._get_store_predictions_callback(self.trainer, key=key)

            # run the trainer predict method, model.forward() is called
            # for each batch and store into the callback cache
            if store_results_callback.is_written:
                logger.info(
                    f"Loading pre-computed vectors from {store_results_callback.cache_file}"
                )
            else:
                trainer.predict(model=self.model, dataloaders=loader, return_predictions=False)

            # return an iterator over the cached batches
            output_batches = store_results_callback.iter_batches()
        else:
            if progress_bar:
                loader = track(iter(loader), description="Processing dataset")

            # process each batch sequentially using `self.process`
            output_batches = map(self.process, loader)

        final_outputs = map(self.postprocess, output_batches)
        for batch in final_outputs:
            yield batch

    def _get_store_predictions_callback(
        self, trainer: Trainer, *, key: str
    ) -> StorePredictionsCallback:
        store_results_callbacks = list(
            filter(lambda c: isinstance(c, StorePredictionsCallback), trainer.callbacks)
        )
        msg = f"Callback of type {type(StorePredictionsCallback)} must be provided."
        assert len(store_results_callbacks) > 0, msg
        if len(store_results_callbacks) == 1:
            store_results_callback = store_results_callbacks[0]
        else:
            store_results_callbacks = list(filter(lambda s: s.name == key, store_results_callbacks))
            "One and only one StorePredictionsCallback must be provided with name=='document'"
            assert len(store_results_callbacks) == 1, msg
            store_results_callback = store_results_callbacks[0]
        return store_results_callback

    def get_rename_output_names_pipe(self, inputs: List[str], output: str) -> Pipe:
        """Format the output of the model"""
        return Sequential(
            RenameKeys({key: output for key in inputs}),
            FilterKeys(In([output, IDX_COL])),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_index"] = faiss.serialize_index(state["_index"])
        return state

    def __setstate__(self, state):
        state["_index"] = faiss.deserialize_index(state["_index"])
        self.__dict__.update(state)

    def _wrap_dataloader(self, loader: DataLoader) -> DataLoader:
        """
        Return a copy of the original dataset such that the `dataset` and `collate_fn` are
        updated such as to return the field `__idx__` along the original data.
        """
        if isinstance(loader.dataset, AddRowIdx):
            return loader

        def exclude(k):
            excluded_values = ["dataset", "collate_fn", "sampler", "batch_sampler"]
            return k in excluded_values or k.startswith("_")

        args = {k: v for k, v in loader.__dict__.items() if not exclude(k)}
        return DataLoader(
            dataset=self._wrap_dataset(loader.dataset),
            collate_fn=self._wrap_collate_fn(loader.collate_fn),
            **args,
        )

    def _wrap_collate_fn(self, collate_fn: Callable) -> Pipe:
        """Wrap the collate_fn to return IDX_COL along the batch values"""
        return Parallel(collate_fn, Collate(IDX_COL))

    def _wrap_dataset(self, dataset: Dataset) -> TorchDataset:
        """Wrap the dataset to return IDX_COL along the batch values"""
        return AddRowIdx(dataset)
