from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pytorch_lightning as pl
import rich
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split

from fz_openqa.datamodules.builders.base import DatasetBuilder
from fz_openqa.datamodules.builders.corpus import CorpusBuilder
from fz_openqa.datamodules.builders.utils.format_row import format_row_flat_questions
from fz_openqa.datamodules.builders.utils.format_row import format_row_nested_questions
from fz_openqa.datamodules.pipelines.collate.field import CollateField
from fz_openqa.datamodules.pipes import BlockSequential
from fz_openqa.datamodules.pipes import Flatten
from fz_openqa.datamodules.pipes import Parallel
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes.inverse_cloze_task import InverseClozeTask
from fz_openqa.datamodules.utils.dataset import get_column_names
from fz_openqa.datamodules.utils.dataset import keep_only_columns
from fz_openqa.datamodules.utils.datastruct import OpenQaDataset
from fz_openqa.datamodules.utils.typing import HfDataset

logger = logging.getLogger(__name__)


class SelectMode(Enum):
    FIRST = "first"
    SAMPLE = "sample"


class InverseClozeTaskBuilder(DatasetBuilder):
    _column_names = [
        "document.row_idx",
        "document.retrieval_score",
        "document.match_score",
    ]

    _pt_attributes = [
        "question.input_ids",
        "question.attention_mask",
        "document.input_ids",
        "document.attention_mask",
        "document.match_score",
        "document.retrieval_score",
    ]

    def __init__(
        self,
        *,
        corpus_builder: CorpusBuilder,
        document_key: str = "document.idx",
        passage_key: str = "document.passage_idx",
        min_distance: int = 1,
        poisson_lambda: float = 2.0,
        n_neighbours: int = 1,
        train_ratio: float = 0.9,
        select_mode: str = "first",
        num_proc: int = 2,
        batch_size: int = 1000,
        writer_batch_size: int = 1000,
        output_columns: Optional[List[str]] = None,
        transform: Optional[Pipe] = None,
        **kwargs,
    ):
        super(InverseClozeTaskBuilder, self).__init__(cache_dir=None, **kwargs)

        # sub-builders
        self.corpus_builder = corpus_builder

        # get the tokenizer
        self.tokenizer = corpus_builder.tokenizer

        # transform for the collate_fn
        self.transform = transform

        # arguments
        self.train_ratio = train_ratio
        self.output_columns = output_columns
        self.n_neighbours = n_neighbours
        self.select_mode = SelectMode(select_mode)
        self.map_args = {
            "min_distance": min_distance,
            "poisson_lambda": poisson_lambda,
            "n_neighbours": self.n_neighbours,
            "passage_key": passage_key,
            "document_key": document_key,
            "num_proc": num_proc,
            "batch_size": batch_size,
            "writer_batch_size": writer_batch_size,
        }

    @property
    def column_names(self):
        return list(set(self._column_names + self.corpus_builder.column_names))

    @property
    def pt_attributes(self):
        attrs = self.corpus_builder.pt_attributes + self._pt_attributes
        return list(set(attrs))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corpus_builder": self.corpus_builder,
            **self.map_args,
        }

    def __repr__(self):
        args = json.dumps({k: str(v) for k, v in self.to_dict().items()}, indent=2)
        return f"{self.__class__.__name__}({args})"

    def _call(
        self,
        format: Optional[str] = "torch",
        columns: Optional[List[str]] = None,
        model: Optional[pl.LightningModule] = None,
        trainer: Optional[pl.Trainer] = None,
        **kwargs,
    ) -> DatasetDict:
        """
        Build the OpenQA dataset using a base `dataset`, which questions are
        mapped to a `corpus` using the `index`.

        Parameters
        ----------
        format
            The format of the dataset (see `Dataset.set_format`)
        columns
            The columns to include in the output dataset.
        model
            The model to use for mapping.
        trainer
            The trainer to use for mapping.
        kwargs
            Additional arguments, pass to the Dataset.map() in `map_dataset`
        Returns
        -------
        OpenQaDataset
            The `dataset` mapped to the `corpus` using the `index`

        """
        # de-activate formatting for the dataset to avoid messing up
        # with the newly added columns in map_dataset
        corpus = self.corpus_builder(format=None)

        ict_dataset: Dataset = self.build_ict(corpus=corpus, **self.map_args, **kwargs)

        # split train/validation/test
        ict_dataset: DatasetDict = ict_dataset.train_test_split(train_size=self.train_ratio)
        ict_dataset[Split.VALIDATION] = ict_dataset[Split.TEST]

        # remove columns that are not needed
        columns = columns or self.output_columns
        if columns is not None:
            ict_dataset = keep_only_columns(ict_dataset, columns=columns)

        if format is not None:
            ict_dataset = self.set_format(ict_dataset, format=format)

        return ict_dataset

    def set_format(self, dataset: HfDataset, *, format: str = "torch") -> HfDataset:
        pt_cols = [c for c in self.pt_attributes if c in get_column_names(dataset)]
        dataset.set_format(type=format, columns=pt_cols, output_all_columns=True)
        return dataset

    def build_ict(
        self,
        *,
        corpus: Dataset,
        document_key: str = "document.idx",
        passage_key: str = "document.passage_idx",
        min_distance: float,
        poisson_lambda: float,
        n_neighbours: int,
        num_proc: int,
        batch_size: int,
        **map_kwargs,
    ) -> Dataset:
        """
        Build the inverse cloze task dataset.
        """

        pipe = InverseClozeTask(
            document_key=document_key,
            passage_key=passage_key,
            min_distance=min_distance,
            poisson_lambda=poisson_lambda,
            n_neighbours=n_neighbours,
            keys=["input_ids", "attention_mask"],
        )
        corpus = corpus.map(
            pipe,
            batch_size=batch_size,
            batched=True,
            num_proc=num_proc,
            desc="Generating Inverse Cloze Task",
            **map_kwargs,
        )

        return corpus

    def get_collate_pipe(self) -> Pipe:
        """Build a Pipe to transform examples into a Batch."""

        collate = Parallel(
            CollateField("question", tokenizer=self.tokenizer, level=0),
            CollateField(
                "document",
                tokenizer=self.tokenizer,
                level=1,
                include_only=["input_ids", "attention_mask", "document.match_score"],
            ),
        )

        return Sequential(collate, self.transform)

    def format_row(self, row: Dict[str, Any]) -> str:
        """Pretty format a dataset row"""
        args = {"dataset_builder": self.corpus_builder, "tokenizer": self.tokenizer}
        return format_row_flat_questions(row, **args)
