from __future__ import annotations

import json
import logging
from enum import Enum
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import rich
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split

from fz_openqa.datamodules.builders.base import DatasetBuilder
from fz_openqa.datamodules.builders.corpus import CorpusBuilder
from fz_openqa.datamodules.builders.medqa import MedQaBuilder
from fz_openqa.datamodules.builders.utils.format_row import format_row_flat_questions
from fz_openqa.datamodules.builders.utils.format_row import format_row_nested_questions
from fz_openqa.datamodules.index import FaissIndex
from fz_openqa.datamodules.index import Index
from fz_openqa.datamodules.index.builder import IndexBuilder
from fz_openqa.datamodules.index.pipes import FetchNestedDocuments
from fz_openqa.datamodules.index.pipes import SearchCorpus
from fz_openqa.datamodules.pipelines.collate.field import CollateField
from fz_openqa.datamodules.pipelines.preprocessing import FetchAndClassifyDocuments
from fz_openqa.datamodules.pipelines.preprocessing import SortDocuments
from fz_openqa.datamodules.pipes import BlockSequential
from fz_openqa.datamodules.pipes import Flatten
from fz_openqa.datamodules.pipes import Parallel
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import RelevanceClassifier
from fz_openqa.datamodules.pipes import SelectDocs
from fz_openqa.datamodules.utils.dataset import filter_questions_by_pos_docs
from fz_openqa.datamodules.utils.dataset import format_size_difference
from fz_openqa.datamodules.utils.dataset import get_column_names
from fz_openqa.datamodules.utils.dataset import keep_only_columns
from fz_openqa.datamodules.utils.datastruct import OpenQaDataset
from fz_openqa.datamodules.utils.map_with_fingerprint import MapWithFingerprint
from fz_openqa.datamodules.utils.typing import HfDataset

logger = logging.getLogger(__name__)


class SelectMode(Enum):
    FIRST = "first"
    SAMPLE = "sample"


class OpenQaBuilder(DatasetBuilder):
    _column_names = [
        "document.row_idx",
        "document.retrieval_score",
        "document.match_score",
    ]

    _pt_attributes = [
        "document.match_score",
        "document.retrieval_score",
    ]

    def __init__(
        self,
        *,
        dataset_builder: MedQaBuilder,
        corpus_builder: CorpusBuilder,
        index_builder: IndexBuilder,
        relevance_classifier: RelevanceClassifier,
        n_retrieved_documents: int,
        n_documents: Optional[Union[int, Dict]] = None,
        max_pos_docs: Optional[int] = None,
        filter_unmatched: bool = True,
        select_mode: str = "first",
        num_proc: int = 2,
        batch_size: int = 100,
        output_columns: Optional[List[str]] = None,
        **kwargs,
    ):
        super(OpenQaBuilder, self).__init__(cache_dir=None, **kwargs)

        # sub-builders
        self.dataset_builder = dataset_builder
        self.corpus_builder = corpus_builder

        # get the tokenizer
        self.tokenizer = dataset_builder.tokenizer
        assert self.tokenizer.vocab == corpus_builder.tokenizer.vocab

        # objects
        self.index_builder = index_builder

        # arguments
        self.output_columns = output_columns
        self.n_documents = n_documents or n_retrieved_documents
        self.max_pos_docs = max_pos_docs
        self.select_mode = SelectMode(select_mode)
        self.map_args = {
            "relevance_classifier": relevance_classifier,
            "n_retrieved_documents": n_retrieved_documents,
            "n_documents": self.n_documents,
            "max_pos_docs": max_pos_docs,
            "filter_unmatched": filter_unmatched,
            "num_proc": num_proc,
            "batch_size": batch_size,
        }

    @property
    def column_names(self):
        return self._column_names + self.dataset_builder.column_names

    @property
    def pt_attributes(self):
        attrs = (
            self.dataset_builder.pt_attributes
            + self.corpus_builder.pt_attributes
            + self._pt_attributes
        )
        return list(set(attrs))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_builder": self.dataset_builder,
            "corpus_builder": self.corpus_builder,
            "index_builder": self.index_builder,
            **self.map_args,
        }

    def __repr__(self):
        args = json.dumps({k: str(v) for k, v in self.to_dict().items()}, indent=2)
        return f"{self.__class__.__name__}({args})"

    def _call(
        self, format: Optional[str] = "torch", columns: Optional[List[str]] = None, **kwargs
    ) -> OpenQaDataset:
        """
        Build the OpenQA dataset using a base `dataset`, which questions are
        mapped to a `corpus` using the `index`.

        Parameters
        ----------
        format
            The format of the dataset (see `Dataset.set_format`)
        columns
            The columns to include in the output dataset.
        kwargs
            Additional arguments, unused here.
        Returns
        -------
        OpenQaDataset
            The `dataset` mapped to the `corpus` using the `index`

        """
        # de-activate formatting for the dataset to avoid messing up
        # with the newly added columns in map_dataset
        dataset = self.dataset_builder(format=None)
        corpus = self.corpus_builder()

        index = self.index_builder(dataset=corpus, **kwargs)

        dataset = self.map_dataset(dataset=dataset, corpus=corpus, index=index, **self.map_args)

        if format is not None:
            dataset = self.set_format(dataset, format=format)

        # remove columns that are not needed

        if columns is not None:
            dataset = keep_only_columns(dataset, columns=columns)
            corpus = keep_only_columns(corpus, columns=columns)

        return OpenQaDataset(dataset=dataset, corpus=corpus, index=index)

    def set_format(self, dataset: HfDataset, *, format: str = "torch") -> HfDataset:
        pt_cols = [c for c in self.pt_attributes if c in get_column_names(dataset)]
        dataset.set_format(type=format, columns=pt_cols, output_all_columns=True)
        return dataset

    def map_dataset(
        self,
        *,
        dataset: DatasetDict,
        corpus: Dataset,
        index: Index,
        n_retrieved_documents: int,
        n_documents: int,
        max_pos_docs: Optional[int],
        num_proc: int,
        batch_size: int,
        relevance_classifier: RelevanceClassifier,
        filter_unmatched: bool,
    ) -> DatasetDict:
        """
        Map the dataset with documents from the corpus.

        NB: SystemExit: 15: is due to an error in huggingface dataset when attempting
        deleting the the dataset, see issue #114.
        """

        question_nesting_level = self.dataset_builder.nesting_level
        document_nesting_level = self.dataset_builder.nesting_level + 1

        map_kwargs = {
            "num_proc": num_proc,
            "batch_size": batch_size,
            "batched": True,
        }

        # cache the dataset using the index's model
        # for nested datasets, the dataset is flatten, so the index
        # in the flatten dataset corresponds to the flattened index
        # in the call of SearchCorpus.
        if isinstance(index, FaissIndex):
            collate_fn = CollateField(
                "question",
                tokenizer=self.tokenizer,
                exclude=["metamap", "text"],
                level=0,
            )

            flat_dataset = self.flatten_dataset(
                dataset,
                level=question_nesting_level,
                keys=["question.input_ids", "question.attention_mask"],
                desc="Flattening dataset before caching",
                **map_kwargs,
            )

            rich.print(f"[green]flat_dataset: {flat_dataset['train']._fingerprint}")

            index.cache_query_dataset(flat_dataset, collate_fn=collate_fn)

        # Search the document and tag them with `document.match_score`
        pipe = BlockSequential(
            [
                (
                    "Search documents",
                    SearchCorpus(
                        index,
                        k=n_retrieved_documents,
                        level=question_nesting_level,
                    ),
                ),
                (
                    "Classify documents",
                    FetchAndClassifyDocuments(
                        corpus_dataset=corpus,
                        classifier=relevance_classifier,
                        level=document_nesting_level,
                        axis=document_nesting_level,
                        n=n_retrieved_documents,
                        extract_gold=question_nesting_level == 0,
                    ),
                ),
                ("Sort documents", SortDocuments(level=document_nesting_level)),
            ]
        )

        # process the dataset with each block
        original_size = {k: len(dset) for k, dset in dataset.items()}
        for k, block in pipe.blocks.items():
            logger.info(f"Processing: {k}")
            mapper = MapWithFingerprint(
                block,
                cache_dir=self.dataset_builder.cache_dir,
                **map_kwargs,
                desc=f"[Mapping] {k}",
            )
            dataset = mapper(dataset)

        # filter out questions that are not match to any  positive document
        if filter_unmatched:

            def fn(split: Split):
                return partial(
                    filter_questions_by_pos_docs,
                    n_documents=n_documents,
                    max_pos_docs=max_pos_docs,
                    split=split,
                    level=question_nesting_level,
                )

            dataset = DatasetDict(
                {
                    split: dset.filter(fn(split), num_proc=num_proc)
                    for split, dset in dataset.items()
                }
            )

        # print the difference in length for each split
        logger.info(format_size_difference(original_size, dataset))

        return dataset

    def flatten_dataset(
        self,
        dataset: Dataset | DatasetDict,
        *,
        level: int = 0,
        keys: List[str] = None,
        **map_kwargs,
    ) -> Dataset:

        dataset = keep_only_columns(dataset, keys)
        if level == 0:
            return dataset

        return dataset.map(Flatten(level=level), **map_kwargs)

    def get_collate_pipe(self) -> BlockSequential:
        """Build a Pipe to transform examples into a Batch."""

        document_nesting_level = self.dataset_builder.nesting_level + 1

        # A. Collate all attributes stored in `self.dataset`
        # document attributes are collated at level 0
        collate_qad = Parallel(
            CollateField(
                "document",
                level=document_nesting_level,
                to_tensor=["match_score", "retrieval_score"],
                id="collate-nested-document-attributes",
            ),
            self.dataset_builder.get_collate_pipe(),
        )

        # B. select documents (resample the field `document.row_idx`)
        select_documents = self.get_select_documents_pipe(
            self.n_documents,
            max_pos_docs=self.max_pos_docs,
            level=document_nesting_level,
            select_mode=self.select_mode.value,
            shuffle=False,
        )

        # C. fetch documents attributes from `self.corpus` (e.g. document.input_ids, document.text)
        fetch_documents = FetchNestedDocuments(
            corpus_dataset=self.corpus_builder(columns=self.output_columns),
            collate_pipe=self.corpus_builder.get_collate_pipe(),
            level=document_nesting_level,
        )

        return BlockSequential(
            [
                ("Collate Q&A + document indexes", collate_qad),
                ("Select documents", select_documents),
                ("Fetch document data", fetch_documents),
            ],
            id="collate-pipeline",
        )

    @staticmethod
    def get_select_documents_pipe(
        n_documents: Union[int, Dict],
        *,
        max_pos_docs: Optional[int],
        level: int = 1,
        select_mode: str = "first",
        shuffle: bool = False,
    ) -> Optional[Pipe]:
        if n_documents == 0:
            return None

        return SelectDocs(
            total=n_documents,
            max_pos_docs=max_pos_docs,
            pos_select_mode=select_mode,
            neg_select_mode=select_mode,
            strict=False,
            update=True,
            level=level,
            shuffle=shuffle,
        )

    def format_row(self, row: Dict[str, Any]) -> str:
        """Pretty format a dataset row"""

        args = {"dataset_builder": self.dataset_builder, "tokenizer": self.tokenizer}
        if self.dataset_builder.nesting_level == 0:
            return format_row_flat_questions(row, **args)
        elif self.dataset_builder.nesting_level == 1:
            return format_row_nested_questions(row, **args)
        else:
            raise ValueError(f"Unsupported nesting level: {self.dataset_builder.nesting_level}")
