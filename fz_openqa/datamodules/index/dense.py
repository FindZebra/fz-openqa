from __future__ import annotations

import logging
import os.path
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import faiss.contrib.torch_utils  # type: ignore
import numpy as np
import pyarrow as pa
import pytorch_lightning as pl
import torch
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split
from faiss.swigfaiss import Index as FaissSwigIndex
from pytorch_lightning import Trainer
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

from fz_openqa.datamodules.index.base import camel_to_snake
from fz_openqa.datamodules.index.base import Index
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.datamodules.pipes.base import Pipe
from fz_openqa.datamodules.pipes.collate import Collate
from fz_openqa.datamodules.pipes.predict import DEFAULT_LOADER_KWARGS
from fz_openqa.datamodules.pipes.predict import Predict
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import OutputFormat
from fz_openqa.utils.datastruct import PathLike
from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.utils.functional import infer_batch_size
from fz_openqa.utils.tensor_arrow import TensorArrowTable

logger = logging.getLogger(__name__)


def display_file_size(key: str, fn: PathLike, print_fn=None):
    if print_fn is None:
        print_fn = print
    fn = Path(fn)
    if fn.is_dir():
        print(f"{key}:")
        for f in fn.iterdir():
            display_file_size(f"{f} -- {os.path.basename(fn.name)}", f, print_fn)
    else:
        s = os.path.getsize(fn)
        s /= 1024 ** 3
        msg = f"{key} - disk_size={s:.3f} GB"
        print_fn(msg)


def _memory_mapped_arrow_table_from_file(filename: str) -> pa.Table:
    """see table.py in datasets"""
    memory_mapped_stream = pa.memory_map(filename)
    opened_stream = pa.ipc.open_stream(memory_mapped_stream)
    pa_table = opened_stream.read_all()
    return pa_table


def iter_batches_with_indexes(loader: Generator | DataLoader) -> Iterable[Tuple[List[int], Batch]]:
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
    "factory": "IVF100,PQ16x8",
    "metric_type": faiss.METRIC_INNER_PRODUCT,
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

    Parameters
    ----------
    model
        The model to use for indexing and querying, e.g. a `pl.LightningModule`.
    index_name
        The name of the index. Must be unique for each configuration.
    """

    default_key = ["input_ids", "attention_mask"]
    _index: FaissSwigIndex = None
    model: Callable = None
    _vectors_table: Optional[TensorArrowTable] = None
    _master: bool = True
    _max_num_proc: int = 1
    no_fingerprint: List[str] = Index.no_fingerprint + [
        "_max_sim",
        "_emb2pid",
        "faiss_args",
        "in_memory",
        "loader_kwargs",
        "persist_cache",
        "cache_dir",
        "trainer",
        "progress_bar",
        "predict_docs",
        "predict_queries",
    ]

    _pickle_exclude_list: List[str] = [
        "_index",
        "_emb2pid",
        "_vectors_table",
        "_vectors",
        "_max_sim",
        "_is_gpu",
    ]

    def _prepare_index(
        self,
        *,
        model: pl.LightningModule,
        trainer: Optional[Trainer] = None,
        faiss_args: Dict[str, Any] = None,
        faiss_train_size: int = 1000,
        dtype: str = "float32",
        in_memory: bool = True,
        model_output_keys: List[str],
        loader_kwargs: Optional[Dict] = None,
        collate_pipe: Pipe = None,
        persist_cache: bool = False,
        cache_dir: Optional[str] = None,
        progress_bar: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model
            The model to use for indexing and querying, e.g. a `pl.LightningModule`.
        trainer
            The trainer to use for indexing and querying, e.g. a `pl.Trainer`.
            Indexing and querying can also be done without a trainer.
        faiss_args
            Additional arguments to pass to the Faiss index.
        faiss_train_size
            Number of data points to train the index on.
        dtype:
            The dtype of the vectors.
        in_memory:
            Whether to store the index in memory or load using pyarrow (only for Colbert)
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

        # model and params
        self.model = model
        self.faiss_args = faiss_args
        self.dtype = dtype
        self.faiss_train_size = faiss_train_size
        self.in_memory = in_memory
        self.trainer = trainer

        # column names
        self.model_output_keys = model_output_keys

        # trainer and dataloader
        self.trainer = trainer
        self.loader_kwargs = loader_kwargs or DEFAULT_LOADER_KWARGS
        self.progress_bar = progress_bar

        # collate pipe use to convert dataset rows into a batch
        self.collate = collate_pipe or Collate()

        # process pipe, used to process a batch with the model
        # warning: this is actually not used when using the trainer
        self.persist_cache = persist_cache
        if persist_cache is False:
            cache_dir = tempfile.mkdtemp(dir=cache_dir)
        self.cache_dir = cache_dir
        self.predict_docs = Predict(
            self.model, model_output_keys=model_output_keys, output_dtype=self.dtype
        )
        self.predict_queries = Predict(
            self.model, model_output_keys=model_output_keys, output_dtype=self.dtype
        )

    def build(self, dataset: Dataset, **kwargs):
        """
        Build and cache the index. Cache is skipped if `name` or `cache_dir` is not provided.

        Parameters
        ----------
        dataset
            The dataset to index.
        kwargs
            Additional arguments to pass to `_build`
        Returns
        -------
        None

        """

        self._set_index_name(dataset=dataset)

        if self.index_file.exists():
            self._read_index()
        else:
            # build the index
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            self._build(dataset, vector_file=self.vector_file, **kwargs)
            self._write_index()

        # display file sizes
        display_file_size("index", self.index_file, print_fn=logger.info)
        display_file_size("vectors", self.vector_file, print_fn=logger.info)

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

        # if a trainer is not available, use the default one
        trainer = self.trainer
        if trainer is None:
            logger.warning("No trainer provided. Using default loader.")
            trainer = Trainer(
                checkpoint_callback=False,
                logger=False,
                progress_bar_refresh_rate=int(self.progress_bar),
            )

        # process the dataset using the model and cache the vectors
        self._cache_vectors(
            dataset,
            predict=self.predict_docs,
            trainer=trainer,
            collate_fn=self.collate,
            target_file=self.vector_file,
            persist=True,
        )

        # read the vectors from the cache as a pyarrow table
        cached_vectors: TensorArrowTable = self._read_vectors_table()

        # load the vectors form the table
        it = iter(cached_vectors.to_batches(self.faiss_train_size))

        # define the progress bar
        train_size = min(cached_vectors.num_rows, self.faiss_train_size)
        logger.info(f"Index training size: {train_size}")

        # init the faiss index and add the 1st batch
        vectors = next(it)
        self._init_index(vectors)
        self._add_batch_to_index(vectors)

        # init the progress bar
        pbar = tqdm(
            total=(cached_vectors.num_rows // train_size) - 1,
            desc=f"Building {type(self).__name__} (batch_size={train_size})",
            disable=not self.progress_bar,
        )

        # iterate through the remaining batches and add them to the index
        while True:
            try:
                vectors = next(it)
                # rich.print(f"> vectors: {vectors.shape}")
                self._add_batch_to_index(vectors)
                if pbar is not None:
                    pbar.update(1)
            except StopIteration:
                break

        if pbar is not None:
            pbar.close()

        self._train_ends()
        logger.info(
            f"Index is_trained={self._index.is_trained}, "
            f"size={self._index.ntotal}, type={type(self._index)}"
        )

    def _cache_vectors(
        self,
        dataset: Dataset | DatasetDict,
        *,
        predict: Predict,
        trainer: Trainer,
        collate_fn: Callable,
        cache_dir: Optional[Path] = None,
        target_file: Optional[PathLike] = None,
        **kwargs,
    ):
        """
        Compute and cache the vectors for the entire dataset.
        Parameters
        ----------
        dataset
            The dataset to cache the vectors for.
        predict
            The pipe to use to compute and cache the vectors.
        trainer
            The trainer to use to compute the vectors.
        collate_fn
            The pipe used to collate rows from the dataset
        target_file
            The path to save the vectors to.

        Returns
        -------
        None
        """
        predict.invalidate_cache()
        predict.cache(
            dataset,
            trainer=trainer,
            collate_fn=collate_fn,
            loader_kwargs=self.loader_kwargs,
            cache_dir=cache_dir,
            target_file=target_file,
            **kwargs,
        )

    def _init_index(self, vectors: Tensor):
        """
        Initialize the index

        Notes
        ----------
        `faiss_args` description:
            n_list
                The number of cells (space partition). Typical value is sqrt(N)
            n_subvectors
                The number of sub-vectors. Typically this is 8, 16, 32, etc.
                Must be a divisor of the dimension
            n_bits
                Bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte.

        """
        # reshape as 2D array
        logger.info(f"Initializing index with vectors: {vectors.shape}")
        vectors = vectors.view((-1, vectors.shape[-1])).contiguous()

        # init the faiss index
        dim = vectors.shape[-1]
        factory_str = self.faiss_args.get("factory", None)
        metric_type = self.faiss_args["metric_type"]
        logger.info(f"Initializing faiss index with {factory_str}")
        if factory_str is not None:
            self._index = faiss.index_factory(dim, factory_str, metric_type)
        else:
            warnings.warn(
                "Building index without specifying a factory is deprecated.", DeprecationWarning
            )
            index_type = self.faiss_args["type"]
            if index_type == "IVFQ":
                n_list = self.faiss_args["n_list"]
                n_subvectors = self.faiss_args["n_subvectors"]
                n_bits = self.faiss_args["n_bits"]

                assert dim % n_subvectors == 0, "m must be a divisor of dim"
                quantizer = faiss.IndexFlatL2(dim)
                self._index = faiss.IndexIVFPQ(
                    quantizer, dim, n_list, n_subvectors, n_bits, metric_type
                )
            elif index_type == "flat":
                self._index = faiss.IndexFlat(dim, metric_type)
            else:
                raise ValueError(f"Unknown index type {index_type}")

        # move index to GPU if available
        n_gpus = faiss.get_num_gpus()
        if n_gpus > 0:
            logger.info(f"Moving faiss index to GPU n_gpus={n_gpus}")
            self._index = faiss.index_cpu_to_all_gpus(self._index)

        # set n_probe
        self._index.nprobe = self.faiss_args.get("nprobe", 16)

        # train the index once using the 1st batch
        self._train(vectors)

    def _train_ends(self):
        """move the index back to CPU if it was moved to GPU"""
        if isinstance(self._index, faiss.IndexReplicas):
            self._index = faiss.index_gpu_to_cpu(self._index)  # type: ignore

    def _train(self, vectors: torch.Tensor):
        """
        Train faiss index on data

        Parameters
        ----------
        vector
            vector from batch

        Notes
        ----------
        This crashes if using faiss!=1.6.5
        """
        logger.info(f"Training index with {vectors.shape[0]} vectors")

        # todo: check if casting is necessary
        vectors = vectors.to(torch.float32)
        self._index.train(vectors)
        assert self._index.is_trained is True, "Index is not trained"
        logger.info("Index is trained.")

    def _add_batch_to_index(self, vectors: torch.Tensor):
        """
        Add one batch of data to the index
        """

        # add the vectors to the index
        vectors = vectors.to(torch.float32)
        self._index.add(vectors.view(-1, vectors.shape[-1]))

    def _search_chunk(
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
        vector = self._get_vector_from_batch(query)
        score, indices = self.index.search(vector, k)
        return SearchResult(score=score, index=indices, dataset_size=self.dataset_size, k=k)

    def _preprocess_query(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Optional[Split] = None, **kwargs
    ) -> Batch:
        """Preprocess the batch before query"""
        return self.predict_queries(batch, idx=idx, split=split)

    def _call_dataset(
        self,
        dataset: Dataset,
        *,
        k=1,
        collate_fn: Optional[Callable] = None,
        trainer: Optional[Trainer] = None,
        split: Optional[Split] = None,
        persist_cache: bool = False,
        **kwargs,
    ) -> Dataset:
        trainer = trainer or self.trainer
        if trainer:
            self.cache_query_dataset(
                dataset, collate_fn=collate_fn, trainer=trainer, persist_cache=persist_cache
            )

        new_dataset = super()._call_dataset(dataset, k=k, output_format=OutputFormat.LIST, **kwargs)
        if not persist_cache:
            self.predict_queries.delete_cached_files(split=split)
        return new_dataset

    def cache_query_dataset(
        self,
        dataset: Union[Dataset, DatasetDict],
        *,
        collate_fn: Callable,
        trainer: Trainer = None,
        persist_cache: bool = False,
        **kwargs,
    ):
        trainer = trainer or self.trainer
        self._cache_vectors(
            dataset,
            predict=self.predict_queries,
            collate_fn=collate_fn,
            trainer=trainer,
            cache_dir=self.cache_dir,
            persist=persist_cache,
            **kwargs,
        )

    def _get_vector_from_batch(self, batch: Batch) -> np.ndarray:
        """Get and cast the vector from the batch"""
        vector: np.ndarray | List = batch[Predict.output_key]
        return vector

    def _init_loader(self, dataset, *, collate_fn: Callable, **kwargs):
        loader_kwargs = self.loader_kwargs.copy()
        for k, v in kwargs.items():
            loader_kwargs[k] = v

        return Predict.init_loader(
            dataset, collate_fn=collate_fn, loader_kwargs=loader_kwargs, wrap_indices=False
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # state["model"] = None
        state["trainer"] = None
        # state["_index"] = faiss.serialize_index(state["_index"])
        for key in self._pickle_exclude_list:
            if key in state:
                state.pop(key)
        return state

    def __setstate__(self, state):
        # state["_index"] = faiss.deserialize_index(state["_index"])
        state["_master"] = False
        self.__dict__.update(state)

    def __del__(self):
        if isinstance(self._index, faiss.IndexReplicas):
            self._index = faiss.index_gpu_to_cpu(self._index)  # type: ignore
        if self._master and self.persist_cache is False:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    # def _get_fingerprint_struct(self) -> List | Dict:
    #     """Add the model fingerprint to the struct"""
    #     fingerprints = super()._get_fingerprint_struct()
    #     fingerprints["model"] = get_fingerprint(self.model)
    #     return fingerprints

    @property
    def is_indexed(self):
        return self._index is None or self._index.is_trained

    @property
    def index(self) -> FaissIndex:
        if self._index is None:
            self._read_index()
        return self._index  # type: ignore

    @property
    def ntotal(self):
        return self._index is None or self._index.ntotal

    @property
    def index_file(self):
        return Path(self.cache_dir) / "indexes" / f"{self.index_name}" / "index.faiss"

    @property
    def vector_file(self):
        return Path(self.cache_dir) / "indexes" / f"{self.index_name}" / "vectors.tsarrow"

    def _write_index(self):
        logger.info(f"Writing {type(self).__name__} [to]: {self.index_file.absolute()}")
        faiss.write_index(self._index, str(self.index_file))

    def _read_index(self):
        logger.info(f"Reading {type(self).__name__} from: {self.index_file.absolute()}")
        self._index = faiss.read_index(str(self.index_file))

    def _read_vectors_table(self) -> TensorArrowTable:
        logger.info(f"Reading vectors table from: {self.vector_file.absolute()}")
        return TensorArrowTable(self.vector_file, dtype=self.dtype)

    def _set_index_name(self, dataset) -> None:
        """Set the index name. Must be unique to allow for sage caching."""
        cls_id = camel_to_snake(type(self).__name__)
        pipe_fingerprint = self.fingerprint(reduce=True, exclude=["model"])
        model_fingerprint = get_fingerprint(self.model)
        self.index_name = f"{cls_id}-{dataset._fingerprint}-{pipe_fingerprint}-{model_fingerprint}"
        logger.info(f"Index name: {self.index_name}")
