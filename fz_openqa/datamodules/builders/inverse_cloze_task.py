from __future__ import annotations

import json
import logging
from enum import Enum
from functools import partial
from numbers import Number
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pytorch_lightning as pl
import torch
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split

from fz_openqa.datamodules.builders.base import DatasetBuilder
from fz_openqa.datamodules.builders.corpus import CorpusBuilder
from fz_openqa.datamodules.pipelines.collate.field import CollateField
from fz_openqa.datamodules.pipes import Apply
from fz_openqa.datamodules.pipes import Parallel
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes.inverse_cloze_task import InverseClozeTask
from fz_openqa.datamodules.utils.dataset import get_column_names
from fz_openqa.datamodules.utils.dataset import keep_only_columns
from fz_openqa.datamodules.utils.datastruct import OpenQaDataset
from fz_openqa.datamodules.utils.typing import HfDataset
from fz_openqa.tokenizers.static import DOC_TOKEN
from fz_openqa.tokenizers.static import QUERY_TOKEN
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pretty_decode

logger = logging.getLogger(__name__)


class SelectMode(Enum):
    FIRST = "first"
    SAMPLE = "sample"


def replace_with_n_tokens(x: torch.Tensor | list | Number, *, a: Any, b: Any, n: int, **kwargs):
    """replace the token `a` with `n` `b` tokens"""
    if isinstance(x, list):
        if isinstance(x[0], list):
            x = [replace_with_n_tokens(y, a=a, b=b, n=n, **kwargs) for y in x]
        else:
            idx = x.index(a)
            x = x[:idx] + n * [b] + x[idx + 1 :]
    else:
        raise TypeError(f"Cannot handle input of type {type(x)}")

    return x


def pad_first(x: torch.Tensor | list | Number, *, value: Any, n: int, **kwargs):
    """pad the sequence with `n``value`tokens"""
    if isinstance(x, list):
        if isinstance(x[0], list):
            x = [pad_first(y, value=value, n=n, **kwargs) for y in x]
        else:
            x = n * [value] + x
    else:
        raise TypeError(f"Cannot handle input of type {type(x)}")

    return x


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
        n_query_tokens: int = 10,
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
            "n_query_tokens": n_query_tokens,
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
        n_query_tokens: int,
        num_proc: int,
        batch_size: int,
        **map_kwargs,
    ) -> Dataset:
        """
        Build the inverse cloze task dataset.
        """

        # Generate Inverse Cloze Task data
        ict_pipe = InverseClozeTask(
            document_key=document_key,
            passage_key=passage_key,
            min_distance=min_distance,
            poisson_lambda=poisson_lambda,
            n_neighbours=n_neighbours,
            keys=["input_ids", "attention_mask"],
        )

        # replace [DOC] with [QUERY] tokens and pad the attention mask
        vocab = self.tokenizer.vocab
        doc_token_id = vocab[DOC_TOKEN]
        query_token_id = vocab[QUERY_TOKEN]
        fn_a = partial(replace_with_n_tokens, a=doc_token_id, b=query_token_id, n=n_query_tokens)
        fn_b = partial(pad_first, value=1, n=n_query_tokens - 1)
        replace_tokens_pipe = Apply({"question.input_ids": fn_a, "question.attention_mask": fn_b})

        corpus = corpus.map(
            Sequential(ict_pipe, replace_tokens_pipe),
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
                include_only=["input_ids", "attention_mask", "match_score", "retrieval_score"],
                to_tensor=["match_score", "retrieval_score"],
            ),
        )

        return Sequential(collate, self.transform)

    def format_row(self, row: Dict[str, Any]) -> str:
        """Pretty format a dataset row"""
        decode_kwargs = {
            "skip_special_tokens": False,
            "tokenizer": self.tokenizer,
        }
        repr = "* Question:"
        repr += (
            pretty_decode(
                row["question.input_ids"],
                **decode_kwargs,
                style="deep_sky_blue3",
            )
            + "\n"
        )

        repr += get_separator("-") + "\n"
        n_docs = len(row["document.input_ids"])
        match_scores = row.get("document.match_score", None)
        doc_scores = row.get("document.retrieval_score", None)
        if doc_scores is None:
            doc_scores = [None] * n_docs
        if match_scores is None:
            match_scores = [None] * n_docs
            n_positives = None
            n_negative = None
        else:
            n_positives = sum(match_scores > 0)
            n_negative = sum(match_scores == 0)
        repr += (
            f"* Documents: n={n_docs}, " f"n_positive={n_positives}, " f"n_negative={n_negative}\n"
        )
        for j in range(min(n_docs, 3)):
            repr += get_separator(".") + "\n"

            match_on = row.get("document.match_on", None)
            match_on = match_on[j] if match_on is not None else None
            repr += (
                f"|-* Document #{1 + j}, "
                f"score={doc_scores[j]}, "
                f"match_score={match_scores[j]}, "
                f"match_on={match_on}\n"
            )

            repr += (
                pretty_decode(
                    row["document.input_ids"][j],
                    **decode_kwargs,
                    style="white",
                )
                + "\n"
            )

        return repr
