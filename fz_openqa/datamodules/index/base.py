from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import T

import datasets
import pytorch_lightning as pl
import rich
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split
from loguru import logger
from pytorch_lightning import Trainer
from torch import nn

from fz_openqa.datamodules.index.engines import AutoEngine
from fz_openqa.datamodules.index.engines.base import IndexEngine
from fz_openqa.datamodules.index.engines.faiss import FaissEngine
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import Predict
from fz_openqa.datamodules.pipes.control.condition import Condition
from fz_openqa.datamodules.pipes.control.condition import HasPrefix
from fz_openqa.datamodules.pipes.control.condition import Reduce
from fz_openqa.datamodules.pipes.predict import DEFAULT_LOADER_KWARGS
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import OutputFormat
from fz_openqa.utils.datastruct import PathLike
from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.utils.tensor_arrow import TensorArrowTable


class Index(Pipe):
    """Keep an index of a Dataset and search it using queries."""

    index_name: Optional[str] = None
    is_indexed: bool = False
    default_key: Optional[str | List[str]] = None
    no_fingerprint: List[str] = [
        "cache_dir",
        "trainer",
        "loader_kwargs",
    ]

    def __init__(
        self,
        corpus: Dataset,
        *,
        engines: List[IndexEngine | Dict] = None,
        query_field: str = "question",
        index_field: str = "document",
        model: pl.LightningModule | nn.Module = None,
        trainer: Optional[Trainer] = None,
        persist_cache: bool = False,
        cache_dir: PathLike = None,
        # Pipe args
        input_filter: Optional[Condition] = None,
        update: bool = False,
        # Argument for computing the vectors
        dtype: str = "float32",
        model_output_keys: List[str] = None,
        loader_kwargs: Optional[Dict] = None,
        corpus_collate_pipe: Pipe = None,
        dataset_collate_pipe: Pipe = None,
        **_,
    ):
        if cache_dir is None:
            cache_dir = datasets.cached_path("./index")

        if model_output_keys is None:
            model_output_keys = ["_hq_", "_hd_"]

        # set the path where to store the index
        if not persist_cache:
            cache_dir = tempfile.TemporaryDirectory(dir=cache_dir)
        self.cache_dir = cache_dir / str(corpus._fingerprint)

        # register the Engines
        self.engines = []
        for engine in engines:
            if isinstance(engine, dict):
                engine = AutoEngine(**engine, path=self.cache_dir, set_unique_path=True)
                self.engines.append(engine)

        # input fields and input filter for query time
        self.query_field = query_field
        self.index_field = index_field
        query_input_filter = HasPrefix(self.query_field)
        if input_filter is not None:
            query_input_filter = Reduce(query_input_filter, input_filter)

        # build the Pipe
        super().__init__(
            input_filter=query_input_filter,
            update=update,
            id=type(self).__name__,
        )

        # Register the model and the pipes used
        # to handle the processing of the data
        assert dtype in {"float32", "float16"}
        self.dtype = dtype
        self.predict_docs = Predict(
            model, model_output_keys=model_output_keys, output_dtype=self.dtype
        )
        self.predict_queries = Predict(
            model, model_output_keys=model_output_keys, output_dtype=self.dtype
        )
        # trainer and dataloader
        self.trainer = trainer
        self.loader_kwargs = loader_kwargs or DEFAULT_LOADER_KWARGS
        # collate pipe use to convert dataset rows into a batch
        self.corpus_collate_pipe = corpus_collate_pipe or Collate()
        self.dataset_collate_pipe = dataset_collate_pipe or Collate()

        # build the engines and save them to disk
        self.build_engines(corpus)

    def build_engines(self, corpus):
        if self.requires_vector:
            vectors = self.cache_vectors(
                corpus,
                predict=self.predict_docs,
                trainer=self.trainer,
                collate_fn=self.corpus_collate_pipe,
                target_file=self.vector_file(self.predict_docs.model, field="corpus"),
                persist=True,
            )
        else:
            vectors = None

        for engine in self.engines:
            engine.build(corpus=corpus, vectors=vectors)
            engine.free_memory()

    @property
    def requires_vector(self):
        DenseIndexes = (FaissEngine,)
        return any(isinstance(engine, DenseIndexes) for engine in self.engines)

    def load(self):
        """Load the index from disk."""
        for engine in self.engines:
            engine.load()

    def _call_batch(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:

        if self.requires_vector:
            batch_ = self.dataset_collate_pipe(batch)
            vectors = self.predict_queries(batch_)[self.predict_queries.output_key]
        else:
            vectors = None

        for engine in self.engines:
            batch = engine(batch, vectors=vectors, **kwargs)

        return batch

    def _call_dataset(self, dataset: Dataset, **kwargs) -> Dataset:
        return self._call_dataset_any(dataset, **kwargs)

    def _call_dataset_dict(self, dataset: DatasetDict, **kwargs) -> DatasetDict:
        return self._call_dataset_any(dataset, **kwargs)

    def _call_dataset_any(self, dataset: T, **kwargs) -> T:
        # cache the query vectors
        if self.requires_vector:
            vectors = self.cache_vectors(
                dataset,
                predict=self.predict_queries,
                trainer=self.trainer,
                collate_fn=self.dataset_collate_pipe,
                target_file=self.vector_file(self.predict_docs.model, field="dataset"),
                persist=True,
            )
            # put the model back to cpu to save memory
            self.predict_queries.model.cpu()
        else:
            vectors = None if isinstance(dataset, Dataset) else {}

        # process the dataset with the Engines
        for engine in self.engines:
            # prepare the engine: load and place to CUDA if needed
            if not engine.is_up:
                engine.load()
            engine.cuda()

            # process the dataset
            if isinstance(dataset, DatasetDict):
                dataset = DatasetDict(
                    {
                        split: engine(
                            dset,
                            vectors=vectors.get(split, None),
                            output_format=OutputFormat.NUMPY,
                            **kwargs,
                        )
                        for split, dset in dataset.items()
                    }
                )
            elif isinstance(dataset, Dataset):
                dataset = engine(
                    dataset, vectors=vectors, output_format=OutputFormat.NUMPY, **kwargs
                )
            else:
                raise TypeError(f"Unsupported dataset type: {type(dataset)}")

            # free the memory allocated by the engine
            engine.free_memory()

        # cleanup the vectors
        if self.requires_vector:
            self.predict_queries.delete_cached_files()

        return dataset

    def cache_vectors(
        self,
        dataset: Dataset | DatasetDict,
        *,
        predict: Predict,
        trainer: Trainer,
        collate_fn: Callable,
        cache_dir: Optional[Path] = None,
        target_file: Optional[PathLike] = None,
        **kwargs,
    ) -> TensorArrowTable | Dict[Split, TensorArrowTable]:
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

        if isinstance(dataset, Dataset):
            predict.cache(
                dataset,
                trainer=trainer,
                collate_fn=collate_fn,
                loader_kwargs=self.loader_kwargs,
                cache_dir=cache_dir,
                target_file=target_file,
                **kwargs,
            )
            return self.read_vectors_table(target_file)
        elif isinstance(dataset, DatasetDict):
            vectors = {}
            for split, dset in dataset.items():
                target_file_split = target_file / str(split)
                predict.cache(
                    dset,
                    trainer=trainer,
                    collate_fn=collate_fn,
                    loader_kwargs=self.loader_kwargs,
                    cache_dir=cache_dir,
                    target_file=target_file_split,
                    **kwargs,
                )
                vectors[split] = self.read_vectors_table(target_file_split)

            return vectors
        else:
            raise TypeError(f"Unknown dataset type {type(dataset)}")

    def vector_file(self, model, field: str = "corpus"):
        model_fingerprint = get_fingerprint(model)
        return Path(self.cache_dir) / "vectors" / f"vectors-{model_fingerprint}-{field}.tsarrow"

    def read_vectors_table(self, vector_file: Path) -> TensorArrowTable:
        logger.info(f"Reading vectors table from: {vector_file.absolute()}")
        return TensorArrowTable(vector_file, dtype=self.dtype)
