from __future__ import annotations

from typing import List
from typing import Optional

import pytorch_lightning as pl
from datasets import Dataset
from datasets import DatasetDict
from pytorch_lightning import Trainer

from fz_openqa.datamodules.index.engines.base import IndexEngine
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes.control.condition import Condition
from fz_openqa.datamodules.pipes.control.condition import HasPrefix
from fz_openqa.datamodules.pipes.control.condition import Reduce
from fz_openqa.utils.datastruct import PathLike


class Index(Pipe):
    """Keep an index of a Dataset and search it using queries."""

    index_name: Optional[str] = None
    is_indexed: bool = False
    default_key: Optional[str | List[str]] = None
    no_fingerprint: List[str] = ["verbose", "index_name", "max_chunksize", "id", "k"]

    def __init__(
        self,
        corpus: Dataset,
        *,
        engines: List[IndexEngine] = None,
        query_field: str = "question",
        index_field: str = "document",
        model: pl.LightningModule = None,
        trainer: Optional[Trainer] = None,
        persist_vectors: bool = False,
        cache_dir: PathLike = None,
        # Pipe args
        input_filter: Optional[Condition] = None,
        update: bool = False,
        **_
    ):

        # register the Engines
        self.engines = engines

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

        # build the engines and save them to disk
        self.build_engines(corpus)

    def build_engines(self, corpus):
        for engine in self.engines:
            engine.train(corpus)
            engine.add(corpus)
            engine.save()
            engine.free_memory()

    def load(self):
        """Load the index from disk."""
        for engine in self.engines:
            engine.load()

    def _call_dataset(self, dataset: Dataset, **kwargs) -> Dataset:
        for engine in self.engines:
            dataset = engine.index(dataset, **kwargs)

        return dataset

    def _call_dataset_dict(self, dataset: DatasetDict, **kwargs) -> DatasetDict:
        for engine in self.engines:
            dataset = engine.index(dataset, **kwargs)

        return dataset
