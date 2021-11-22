import logging
from functools import singledispatchmethod
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import faiss
import numpy as np
import pytorch_lightning as pl
import rich
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split
from faiss.swigfaiss import Index as FaissSwigIndex
from pytorch_lightning import Trainer
from rich.progress import track
from rich.status import Status
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
from fz_openqa.datamodules.pipes.predict import DEFAULT_LOADER_KWARGS
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.utils.functional import infer_batch_size
from fz_openqa.utils.functional import is_index_contiguous

logger = logging.getLogger(__name__)


def iter_batches_with_indexes(
    loader: Union[Generator, DataLoader]
) -> Iterable[Tuple[List[int], Batch]]:
    """
    Iterate over batches and return a tuple of the batch index and the batch itself.
    """
    i = 0
    it = iter(loader)
    for batch in it:
        bs = infer_batch_size(batch)
        idx = list(range(i, i + bs))
        i += bs
        yield idx, batch


DEFAULT_FAISS_KWARGS = {
    "metric_type": faiss.METRIC_INNER_PRODUCT,
    "n_list": 32,
    "m": 16,
    "n_bits": 8,
}


class FaissIndex(Index):
    """
    A dense index using Faiss. This class allows storing document vectors (one-dimensional)
    into a Faiss index and allows querying the index for similar documents.

    This class allows processing `Dataset` using  a `pl.LighningModule` and
    (optionally) a `pl.Trainer`. This is done automatically when passing a Trainer in the
    constructor. This functionality relies on the `Predict` pipe, which is used to
    handle PyTorch Lightning mechanics. `Predict` works by caching predictions to a
    `pyarrow.Table`. Indexes must be passed (i.e. `pipe(batch, idx=idx)`) at all times to
    enables the `Predict` pipe to work.

    The trainer can also be used to process the queries. In that case either
    1. call `cache_query_dataset` before calling `search` on each batch`
    2. call `search` directly on the query `Dataset`

    # todo: handle temporary cache_dir form here (persist=False)

    Parameters
    ----------
    model
        The model to use for indexing and querying, e.g. a `pl.LightningModule`.
    index_name
        The name of the index. Must be unique for each configuration.
    """

    vectors_column_name = "__vectors__"
    _dtype: np.dtype = np.float32
    _index: FaissSwigIndex = None
    model: Callable = None
    index_name: str

    def __init__(
        self,
        dataset: Dataset,
        *,
        model: pl.LightningModule,
        trainer: Optional[Trainer] = None,
        faiss_args: Dict[str, Any] = None,
        index_key: str = "document.row_idx",
        model_output_keys: List[str],
        loader_kwargs: Optional[Dict] = None,
        collate_pipe: Pipe = None,
        persist_cache: bool = True,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        dataset
            The dataset to index.
        model
            The model to use for indexing and querying, e.g. a `pl.LightningModule`.
        trainer
            The trainer to use for indexing and querying, e.g. a `pl.Trainer`.
            Indexing and querying can also be done without a trainer.
        faiss_args
            Additional arguments to pass to the Faiss index.
        index_key
            Name of the column in the indexed dataset to use as index.
        model_output_keys
            Names of the accepted model output keys (vector to use for indexing).
        loader_kwargs
            Additional arguments to pass to the `DataLoader`.
        collate_pipe
            A pipe to use for collating the examples from the `dataset`
        persist_cache
            Whether to persist the cache to disk when calling `predict.cache`
        cache_dir
            The directory to store the cache in.
        kwargs
            Additional arguments to pass to the `Index` class.
        """
        faiss_args = faiss_args or DEFAULT_FAISS_KWARGS
        #  save fingerprints and define the index name. The index name
        # must be unique for each dataset x model x faiss_args combination
        self._model_fingerprint = get_fingerprint(model)
        self._dataset_fingerprint = dataset._fingerprint
        self._faiss_fingerprint = get_fingerprint(faiss_args)
        self._fingerprint = get_fingerprint(
            {
                "model": self._model_fingerprint,
                "dataset": self._dataset_fingerprint,
                "faiss": self._faiss_fingerprint,
            }
        )
        self.index_name = f"{type(self).__name__}-{self._fingerprint}"

        # model and params
        self.model = model
        self.faiss_args = faiss_args
        self.trainer = trainer

        # column names
        self.index_key = index_key
        self.model_output_keys = model_output_keys

        # trainer and dataloader
        self.trainer = trainer
        self.loader_kwargs = loader_kwargs or DEFAULT_LOADER_KWARGS

        # collate pipe use to convert dataset rows into a batch
        self.collate = collate_pipe or Collate()

        # process pipe, used to process a batch with the model
        # warning: this is actually not used when using the trainer
        self.persist_cache = persist_cache
        self.cache_dir = cache_dir
        self.predict_docs = Predict(self.model)
        self.predict_queries = Predict(self.model)

        # postprocessing: rename the model outputs `model_output_keys` to `vectors_column_name`
        self.postprocess = self.get_rename_output_names_pipe(
            inputs=self.model_output_keys, output=self.vectors_column_name
        )

        # call the super: build the index
        super(FaissIndex, self).__init__(
            dataset=dataset, name=self.index_name, cache_dir=cache_dir, **kwargs
        )

    @property
    def is_indexed(self):
        return self._index is None or self._index.is_trained

    def build(
        self, dataset: Dataset, *, name: Optional[str] = None, cache_dir=Optional[None], **kwargs
    ):
        """
        Build and cache the index. Cache is skipped if `name` or `cache_dir` is not provided.

        Parameters
        ----------
        dataset
            The dataset to index.
        name
            The name of the index (must be unique).
        cache_dir
            The directory to store the cache in.
        kwargs
            Additional arguments to pass to `_build`
        Returns
        -------
        None

        """
        if cache_dir is None or name is None:
            # skip caching if not cache_dir is provided
            logger.info("No cache_dir provided. Building index without caching")
            self._build(dataset, cache_dir=cache_dir, **kwargs)

        else:
            cache_file = Path(cache_dir) / "indexes" / f"{name}.index"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            if cache_file.exists():
                # load the index from the cache
                logger.info(f"Loading FaissIndex from cache: {cache_file}")
                self._index = faiss.read_index(str(cache_file))
            else:
                # build the index
                self._build(dataset, cache_dir=cache_dir, **kwargs)
                logger.info(f"Writing FaissIndex to cache: {cache_file}")
                faiss.write_index(self._index, str(cache_file))

    def _build(self, dataset: Dataset, **kwargs):
        """
        Build the index using the model and the dataset.
        Iterate over each batch and add the vectors to the index.
        If a trainer is provided, predictions (vectors) are computed using the trainer
        and cached to a file, otherwise they are computed on the fly.

        Parameters
        ----------
        dataset
            The dataset to index.
        kwargs
            Additional arguments, not used here

        Returns
        -------
        None
        """

        # if a trainer is available, use it to process and cache the whole dataset
        loader_args = {}
        if self.trainer is not None:
            self._cache_vectors(dataset, predict=self.predict_docs, collate_fn=self.collate)
            loader_args["batch_size"] = 1000

        # instantiate the base data loader
        loader = self._init_loader(dataset, collate_fn=self.collate, **loader_args)

        # process batches: this returns a generator, so batches will be processed
        # as they are consumed in the following loop
        processed_batches = self._process_batches(loader, predict=self.predict_docs)

        # build an iterator over the processed batches
        it = iter_batches_with_indexes(processed_batches)

        # init the faiss index and add the 1st batch
        idx, batch = next(it)
        self._init_index(batch)
        self._add_batch_to_index(batch, idx)

        # iterate through the remaining batches and add them to the index
        while True:
            try:
                idx, batch = next(it)
                self._add_batch_to_index(batch, idx=idx)
            except Exception:
                break

        logger.info(
            f"Index is_trained={self._index.is_trained}, "
            f"size={self._index.ntotal}, type={type(self._index)}"
        )

    def _cache_vectors(
        self, dataset: Union[Dataset, DatasetDict], *, predict: Predict, collate_fn: Callable
    ):
        """
        Compute and cache the vectors for the entire dataset.
        Parameters
        ----------
        dataset
            The dataset to cache the vectors for.
        predict
            The pipe to use to compute and cache the vectors.
        collate_fn
            The pipe used to collate rows from the dataset

        Returns
        -------
        None
        """
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
        Initialize the index

        Parameters
        ----------
        n_list
            The number of cells (space partition). Typical value is sqrt(N)
        m
            The number of sub-vector. Typically this is 8, 16, 32, etc.
            Must be a divisor of the dimension
        n_bits
            Bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte.

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

    def _add_batch_to_index(self, batch: Batch, idx: List[int]):
        """
        Add one batch of data to the index
        """
        self._check_index_consistency(idx)

        # add the vectors to the index
        vector = self._get_vector_from_batch(batch)
        assert isinstance(vector, np.ndarray), f"vector {type(vector)} is not a numpy array"
        assert len(vector.shape) == 2, f"{vector} is not a 2D array"
        self._train(vector)
        self._index.add(vector)

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
        """
        Search the index using a batch of queries. For a single query, the batch is processed
        using the model and the predict pipe.
        For a whole query dataset, you might consider calling `cache_query_dataset` first. In that
        case you must provide the `idx` and `split` arguments (see Warnings).
        This is however handled automatically when calling `search` with a `Dataset` as query.

        Warnings
        --------
        `idx` and `split` must be provided if the queried dataset was cached. Caching is
        performed if calling `cache_query_dataset()` beforehand.

        Parameters
        ----------
        query
            The batch of queries to search for.
        idx
            The indexes to search in.
        split
            The dataset split to search in.
        k
        kwargs

        Returns
        -------
        SearchResult
            The search result for the batch

        """
        query = self.predict_queries(query, idx=idx, split=split)
        query = self.postprocess(query)
        return self._query_index(query, k=k)

    @search.register(Dataset)
    def _search_dataset(
        self, query: Dataset, *, k: int = 1, collate_fn: Callable, **kwargs
    ) -> Iterator[SearchResult]:
        """
        This method is called if `query` is of type Dataset

        todo: refactor to return a dataset instead of an iterator.
        todo: replace `search` with `call` and return search results
              as Batch instead of SearchResult
        """
        if self.trainer:
            self.cache_query_dataset(query, collate_fn=collate_fn)

        loader = self._init_loader(query, collate_fn=collate_fn)
        loader = iter_batches_with_indexes(loader)
        for idx, batch in loader:
            yield self.search(batch, k=k, idx=idx, **kwargs)

    @search.register(DatasetDict)
    def _search_dataset_dict(
        self, query: DatasetDict, *, k: int = 1, **kwargs
    ) -> Dict[Split, Iterator[SearchResult]]:
        """This method is called if `query` is of type DatasetDict"""
        # todo : reimplement
        return {split: self.search(dset, k=k, **kwargs) for split, dset in query.items()}

    def _query_index(self, query: Batch, *, k: int) -> SearchResult:
        """Query the index given a batch of data"""
        vector = self._get_vector_from_batch(query)
        score, indices = self._index.search(vector, k)
        return SearchResult(score=score, index=indices, dataset_size=self.dataset_size)

    def cache_query_dataset(
        self, dataset: Union[Dataset, DatasetDict], *, collate_fn: Callable, **kwargs
    ):
        self._cache_vectors(dataset, predict=self.predict_queries, collate_fn=collate_fn)

    def _get_vector_from_batch(self, batch: Batch) -> np.ndarray:
        """Get and cast the vector from the batch"""
        vector: np.ndarray = batch[self.vectors_column_name]
        vector = vector.astype(self._dtype)
        return vector

    def _init_loader(self, dataset, *, collate_fn: Callable, **kwargs):
        loader_kwargs = self.loader_kwargs.copy()
        for k, v in kwargs.items():
            loader_kwargs[k] = v

        return Predict.init_loader(
            dataset, collate_fn=collate_fn, loader_kwargs=loader_kwargs, wrap_indices=False
        )

    def _process_batches(
        self,
        loader: DataLoader,
        *,
        predict: Predict,
        progress_bar: bool = True,
    ) -> Iterable[Batch]:

        """
        This method iterates over batches, which for each batch:
        1. process the batch using the model (or load from cache)
        2. postprocess the output of the model (renaming keys, filtering keys, etc)

        This function processes batches as they are loaded from the dataloader.
        """

        # Make sue the loader is not shuffled
        assert isinstance(
            loader.sampler, SequentialSampler
        ), "Cannot handle DataLoader with shuffle=True."

        # add a progress bar, assume that vectors are cached if the Trainer was provided
        if progress_bar:
            desc = "Ingest Faiss index"
            if self.trainer is not None:
                desc += " (loading vectors from cache)"
            loader = track(loader, description=desc)

        # wrap the loader to return the indexes along the batch
        loader = iter_batches_with_indexes(loader)

        # process batches sequentially
        for idx, batch in loader:
            batch = predict(batch, idx=idx)
            batch = self.postprocess(batch)
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

    def _check_index_consistency(self, idx):
        """Make sure the new idx are consistent with the `_index`"""
        msg = f"Indexes are not contiguous (i.e. 1, 2, 3, 4),\nindexes={idx}"
        assert is_index_contiguous(idx), msg
        msg = (
            f"The stored index and the indexes are not contiguous, "
            f"\nindex_size={self._index.ntotal}, first_index={idx[0]}"
        )
        assert self._index.ntotal == idx[0], msg
