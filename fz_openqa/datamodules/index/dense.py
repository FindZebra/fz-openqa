from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import faiss.contrib.torch_utils  # type: ignore
import pytorch_lightning as pl
import torch
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split
from loguru import logger
from pytorch_lightning import Trainer

from fz_openqa.datamodules.index._base import camel_to_snake
from fz_openqa.datamodules.index._base import Index
from fz_openqa.datamodules.index.engines.base import IndexEngine
from fz_openqa.datamodules.index.engines.faiss import FaissEngine
from fz_openqa.datamodules.index.engines.index_lookup import LookupEngine
from fz_openqa.datamodules.index.engines.multi_faiss import MultiFaissHandler
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.datamodules.pipes.base import Pipe
from fz_openqa.datamodules.pipes.collate import Collate
from fz_openqa.datamodules.pipes.predict import DEFAULT_LOADER_KWARGS
from fz_openqa.datamodules.pipes.predict import Predict
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import OutputFormat
from fz_openqa.utils.datastruct import PathLike
from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.utils.metric_type import MetricType
from fz_openqa.utils.tensor_arrow import TensorArrowTable


class DenseIndex(Index):
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

    default_key = ["input_ids", "attention_mask", "document_idx"]
    _index: Optional[IndexEngine] = None
    model: Callable | torch.nn.Module = None
    _vectors_table: Optional[TensorArrowTable] = None
    _master: bool = True
    _max_num_proc: int = 1
    no_fingerprint: List[str] = Index.no_fingerprint + [
        "faiss_args",
        "in_memory",
        "loader_kwargs",
        "persist_cache",
        "cache_dir",
        "trainer",
        "progress_bar",
        "predict_docs",
        "predict_queries",
        "keep_faiss_on_cpu",
        "train_faiss_on_cpu",
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
        dtype: str | int = "float32",
        handler: str = "flat",
        index_factory: str = "Flat",
        in_memory: bool = True,
        model_output_keys: List[str],
        loader_kwargs: Optional[Dict] = None,
        collate_pipe: Pipe = None,
        persist_cache: bool = False,
        cache_dir: Optional[str] = None,
        progress_bar: bool = False,
        keep_faiss_on_cpu: bool = False,
        train_faiss_on_cpu: bool = False,
        metric_type: MetricType = MetricType.inner_product,
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
        dtype:
            The dtype of the vectors.
        index_factory
            string used to initialized the faiss index
        hanlder:
            string used to initialize the faiss index handler: `flat` or `multi`
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
        # model and params
        self.model = model
        if dtype == 16 or dtype == 32:
            dtype = f"float{dtype}"
        self.dtype = dtype
        self.index_factory = index_factory
        self.handler = handler
        self.in_memory = in_memory
        self.trainer = trainer
        self.keep_faiss_on_cpu = keep_faiss_on_cpu
        self.train_faiss_on_cpu = train_faiss_on_cpu
        self.metric_type = metric_type

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

    def build(
        self,
        dataset: Dataset,
        nprobe: int = 8,
        faiss_train_size=None,
        faiss_tempmem: int = -1,
        shard_faiss=False,
        **kwargs,
    ):
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
        Handler = {"flat": FaissEngine, "multi": MultiFaissHandler, "lookup": LookupEngine}[
            self.handler
        ]

        self._index = Handler(
            path=self.index_path,
            index_factory=self.index_factory,
            keep_on_cpu=self.keep_faiss_on_cpu,
            train_on_cpu=self.train_faiss_on_cpu,
            nprobe=nprobe,
            faiss_train_size=faiss_train_size,
            faiss_tempmem=faiss_tempmem,
            metric_type=self.metric_type,
        )
        self._cache_vectors_and_build(dataset=dataset, **kwargs)

    def _cache_vectors_and_build(self, dataset: Dataset, **kwargs):
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

        # read the vectors from the cache as a pyarrow table and build the index
        vectors = self._read_vectors_table()
        if vectors.dim() == 2:
            stride = None
        elif vectors.dim() == 3:
            stride = vectors.shape[1]
            vectors = vectors.view(-1, vectors.shape[-1])
        else:
            stride = None
            raise ValueError(f"Invalid vectors shape: {vectors.shape}, expected 2 or 3 dimensions.")

        # retrieve document ids
        if isinstance(self._index, (MultiFaissHandler, LookupEngine)):
            doc_ids = dataset["document.idx"]
            doc_ids = self._expand_seq_length(doc_ids, stride)
        else:
            doc_ids = None

        t_init = time.time()
        self._index.build(vectors=vectors, doc_ids=doc_ids)

        # end building the index
        logger.info(
            f"Built index of size={len(self._index)}, "
            f"type={type(self._index)}, "
            f"build time={time.time() - t_init:.2f}s"
        )
        self._train_ends()

    def _expand_seq_length(self, flat_values: List, stride: int):
        if stride is not None:
            flat_values = [stride * [idx] for idx in flat_values]
            flat_values = [item for sublist in flat_values for item in sublist]
        return flat_values

    def _train_ends(self):
        """move the index back to CPU if it was moved to GPU"""
        self.cpu()

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

        # move index to GPUs
        n_gpus = faiss.get_num_gpus()
        if n_gpus > 0:
            self._index.cuda()

        # search faiss
        vector = self._get_vector_from_batch(query)
        score, indices = self.index(vector, k=k, doc_ids=query.get("question.document_idx", None))
        return SearchResult(score=score, index=indices, dataset_size=self.dataset_size, k=k)

    def _preprocess_query(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Optional[Split] = None, **kwargs
    ) -> Batch:
        """Preprocess the batch before query"""
        return self.predict_queries(batch, idx=idx, split=split, format=OutputFormat.TORCH)

    def _get_vector_from_batch(self, batch: Batch) -> torch.Tensor:
        """Get and cast the vector from the batch"""
        return batch[Predict.output_key]

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
        self.free_memory()
        if self._master and hasattr(self, "persist_cache") and self.persist_cache is False:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def free_memory(self):
        if self._index is not None:
            return self._index.free_memory()

    def cpu(self):
        self._index.cpu()

    @property
    def index(self) -> IndexEngine:
        if not self._index.is_up:
            self._index.load()
        return self._index

    @property
    def ntotal(self):
        return len(self._index)

    @property
    def index_path(self):
        return Path(self.cache_dir) / "indexes" / f"{self.index_name}" / "index"

    @property
    def vector_file(self):
        return Path(self.cache_dir) / "indexes" / f"{self.index_name}" / "vectors.tsarrow"

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
            self.free_memory()
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
