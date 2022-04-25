from __future__ import annotations

import json
from copy import copy
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pytorch_lightning as pl
from datasets import Dataset
from datasets import DatasetDict
from loguru import logger
from omegaconf import DictConfig

from fz_openqa.datamodules.builders.base import DatasetBuilder
from fz_openqa.datamodules.builders.corpus import CorpusBuilder
from fz_openqa.datamodules.builders.qa import QaBuilder
from fz_openqa.datamodules.builders.transforms.base import OpenQaTransform
from fz_openqa.datamodules.builders.utils.format_row import format_row_concatenated_questions
from fz_openqa.datamodules.builders.utils.format_row import format_row_flat_questions
from fz_openqa.datamodules.builders.utils.format_row import format_row_flat_questions_with_docs
from fz_openqa.datamodules.builders.utils.format_row import format_row_nested_questions_with_docs
from fz_openqa.datamodules.index import DenseIndex
from fz_openqa.datamodules.index import Index
from fz_openqa.datamodules.index.builder import IndexBuilder
from fz_openqa.datamodules.index.index_pipes import FetchNestedDocuments
from fz_openqa.datamodules.index.index_pipes import SearchCorpus
from fz_openqa.datamodules.pipelines.collate.field import CollateField
from fz_openqa.datamodules.pipelines.preprocessing import FetchAndClassifyDocuments
from fz_openqa.datamodules.pipelines.preprocessing import SortDocuments
from fz_openqa.datamodules.pipes import BlockSequential
from fz_openqa.datamodules.pipes import Flatten
from fz_openqa.datamodules.pipes import Parallel
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import RelevanceClassifier
from fz_openqa.datamodules.pipes import Sampler
from fz_openqa.datamodules.pipes import ScoreTransform
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes.control.condition import HasPrefix
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.datamodules.pipes.nesting import ApplyAsFlatten
from fz_openqa.datamodules.pipes.nesting import Nested
from fz_openqa.datamodules.utils.dataset import get_column_names
from fz_openqa.datamodules.utils.dataset import keep_only_columns
from fz_openqa.datamodules.utils.datastruct import OpenQaConfig
from fz_openqa.datamodules.utils.datastruct import OpenQaDataset
from fz_openqa.datamodules.utils.map_with_fingerprint import MapWithFingerprint
from fz_openqa.datamodules.utils.typing import HfDataset


class SelectMode(Enum):
    FIRST = "first"
    SAMPLE = "sample"


class OpenQaBuilder(DatasetBuilder):
    _column_names = [
        "document.row_idx",
        "document.retrieval_score",
        "document.match_score",
        "document.retrieval_rank",
    ]

    _pt_attributes = [
        "document.match_score",
        "document.retrieval_score",
        "document.retrieval_rank",
    ]

    def __init__(
        self,
        *,
        dataset_builder: QaBuilder,
        corpus_builder: CorpusBuilder,
        index_builder: IndexBuilder,
        relevance_classifier: Optional[RelevanceClassifier],
        score_transform: Optional[ScoreTransform] = None,
        sampler: Optional[Sampler],
        n_retrieved_documents: int,
        document_nesting_level: Optional[int] = None,
        sort_documents: bool = False,
        dataset_transform: Optional[OpenQaTransform] = None,
        num_proc: int = 2,
        batch_size: int = 100,
        writer_batch_size: int = 1000,
        output_columns: Optional[List[str]] = None,
        transform: Optional[Pipe | DictConfig] = None,
        **kwargs,
    ):
        super(OpenQaBuilder, self).__init__(cache_dir=None, **kwargs)

        # sub-builders
        self.dataset_builder = dataset_builder
        self.corpus_builder = corpus_builder

        # final dataset configuration & nesting levels
        if document_nesting_level is None:
            document_nesting_level = self.dataset_builder.nesting_level + 1
        self._document_base_nesting_level = document_nesting_level

        self.openqa_config: Optional[OpenQaConfig] = OpenQaConfig(
            question_nesting_level=self.dataset_builder.nesting_level,
            document_nesting_level=self._document_base_nesting_level,
        )

        # set transform and update the OpenQaConfig
        if dataset_transform is not None:
            assert isinstance(dataset_transform, OpenQaTransform)
            self.openqa_config = dataset_transform(self.openqa_config)
            self.dataset_transform = dataset_transform
        else:
            self.dataset_transform = None

        # get the tokenizer
        self.tokenizer = dataset_builder.tokenizer
        assert self.tokenizer.vocab == corpus_builder.tokenizer.vocab

        # transform for the collate_fn
        self.transform = transform

        # objects
        self.index_builder = index_builder

        # arguments
        self.output_columns = output_columns
        self.sampler = sampler
        self.map_args = {
            "relevance_classifier": relevance_classifier,
            "score_transform": score_transform,
            "n_retrieved_documents": n_retrieved_documents,
            "sort_documents": sort_documents,
            "num_proc": num_proc,
            "batch_size": batch_size,
            "writer_batch_size": writer_batch_size,
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
        self,
        format: Optional[str] = "torch",
        columns: Optional[List[str]] = None,
        model: Optional[pl.LightningModule] = None,
        trainer: Optional[pl.Trainer] = None,
        splits: Optional[List[str]] = None,
        **kwargs,
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
        dataset = self.dataset_builder(format=None)
        corpus = self.corpus_builder()

        # select splits
        if splits is not None:
            dataset = DatasetDict({k: v for k, v in dataset.items() if k in splits})

        # build the index, potentially using a model
        index = self.index_builder(
            dataset=corpus,
            model=model,
            trainer=trainer,
            collate_pipe=self.corpus_builder._get_collate_pipe(),
        )

        # map the corpus to the dataset
        map_args = copy(self.map_args)
        for k in list(kwargs.keys()):
            if k in map_args:
                map_args[k] = kwargs.pop(k)
        dataset = self.map_dataset(
            dataset=dataset, corpus=corpus, index=index, **map_args, **kwargs
        )

        # transform the dataset
        if self.dataset_transform is not None:
            dataset = self.dataset_transform(
                dataset,
                openqa_config=self.openqa_config,
                **{k: v for k, v in map_args.items() if k in ["num_proc", "batch_size"]},
            )

        # format the dataset
        if format is not None:
            dataset = self.set_format(dataset, format=format)

        # remove columns that are not needed
        columns = columns or self.output_columns
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
        num_proc: int,
        batch_size: int,
        relevance_classifier: Optional[RelevanceClassifier],
        score_transform: Optional[ScoreTransform],
        sort_documents: bool,
        **map_kwargs,
    ) -> DatasetDict:
        """
        Map the dataset with documents from the corpus.

        NB: SystemExit: 15: is due to an error in huggingface dataset when attempting
        deleting the the dataset, see issue #114.
        """
        question_nesting_level = self.dataset_builder.nesting_level

        # cache the dataset using `index.model`, so vectors can be reused in `SearchCorpus`
        # for nested datasets, the dataset is flatten, so the index
        # in the flatten dataset corresponds to the flattened index
        # in the call of SearchCorpus.
        if isinstance(index, DenseIndex):
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
                batched=True,
                num_proc=num_proc,
                batch_size=100,
            )
            index.cache_query_dataset(flat_dataset, collate_fn=collate_fn)

            # move the model back to CPU to save GPU memory
            index.model.cpu()

        # Search the document and tag them with `document.match_score`
        pipe = BlockSequential(
            [
                (
                    "Search documents",
                    SearchCorpus(
                        index,
                        k=n_retrieved_documents,
                        # The search is applied to the flattened dataset,
                        # the right nesting level corresponds is always
                        # the one of the documents minus 1: {question : [doc_1, doc_2, ...]}
                        level=self._document_base_nesting_level - 1,
                    ),
                ),
                (
                    "Classify documents",
                    FetchAndClassifyDocuments(
                        corpus_dataset=corpus,
                        classifier=relevance_classifier,
                        level=self._document_base_nesting_level,
                        axis=self._document_base_nesting_level,
                        n=n_retrieved_documents,
                        # When the documents are nested by one level, only the
                        # gold answer is used to compute the match score. When
                        # the level == 2, the match score is computed using each
                        # answer option, therefore there is no need for extracitng
                        # the gold answer.
                        extract_gold=self._document_base_nesting_level < 2,
                    )
                    if relevance_classifier is not None
                    else None,
                ),
                (
                    "Transform scores",
                    ApplyAsFlatten(
                        score_transform,
                        level=self._document_base_nesting_level - 1,
                        input_filter=In(["document.retrieval_score", "document.match_score"]),
                        update=True,
                    )
                    if score_transform is not None and relevance_classifier is not None
                    else None,
                ),
                (
                    "Sort documents",
                    SortDocuments(level=self._document_base_nesting_level)
                    if sort_documents
                    else None,
                ),
            ]
        )

        # adjust the batch size to account for the documents
        map_kwargs = {
            "num_proc": num_proc,
            "batch_size": batch_size,
            "batched": True,
            # about: writer_batch_size:
            # to avoid: pyarrow.lib.ArrowInvalid: Invalid null value
            #   https://github.com/huggingface/datasets/issues/2831
            # but setting value to high causes the writer to hang:
            #   https://github.com/huggingface/datasets/issues/482
            **map_kwargs,
        }

        for k, block in pipe.blocks.items():
            logger.info(f"Processing: {k}")
            mapper = MapWithFingerprint(
                block,
                cache_dir=self.dataset_builder.cache_dir,
                **map_kwargs,
                desc=f"[Mapping] {k}",
                debug=False,
                id=k,
            )
            dataset = mapper(dataset)

        # free-up GPU memory
        index.free_memory()

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

        # A. Collate all attributes stored in `self.dataset`
        # document attributes are collated at level 0
        collate_qad = Parallel(
            Sequential(
                CollateField(
                    "document",
                    level=self.openqa_config.document_nesting_level,
                    to_tensor=["match_score", "retrieval_score", "retrieval_rank"],
                    id="collate-nested-document-attributes",
                ),
            ),
            self.dataset_builder._get_collate_pipe(
                nesting_level=self.openqa_config.question_nesting_level
            ),
        )

        # B. select documents (resample the field `document.row_idx`)
        select_documents = self.get_select_documents_pipe(
            self.sampler, level=self.openqa_config.document_nesting_level
        )

        # C. fetch documents attributes from `self.corpus` (e.g. document.input_ids, document.text)
        fetch_documents = FetchNestedDocuments(
            corpus_dataset=self.corpus_builder(columns=self.output_columns),
            collate_pipe=self.corpus_builder._get_collate_pipe(),
            level=self.openqa_config.document_nesting_level,
        )

        return BlockSequential(
            [
                ("Collate Q&A + document indexes", collate_qad),
                ("Select documents", select_documents),
                ("Fetch document data", fetch_documents),
                ("Transform", self.transform),
            ],
            id="collate-pipeline",
        )

    @staticmethod
    def get_select_documents_pipe(
        sampler: Optional[Sampler],
        *,
        level: int = 1,
    ) -> Optional[Pipe]:
        if sampler is None:
            return None

        input_filter = HasPrefix(f"{sampler.field}")
        return Nested(pipe=sampler, level=level, input_filter=input_filter, update=True)

    def format_row(self, row: Dict[str, Any], **kwargs) -> str:
        """Pretty format a dataset row

        Parameters
        ----------
        **kwargs
        """

        format_question_fn = {0: format_row_flat_questions, 1: format_row_concatenated_questions}[
            self.openqa_config.question_nesting_level
        ]
        _kwargs = {"format_question_fn": format_question_fn, "tokenizer": self.tokenizer}
        if self.openqa_config.question_nesting_level == 0:
            return format_row_flat_questions_with_docs(row, **_kwargs, **kwargs)
        elif self.openqa_config.question_nesting_level == 1:
            return format_row_nested_questions_with_docs(
                row,
                document_nesting_level=self.openqa_config.document_nesting_level,
                **_kwargs,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported nesting level: {self.openqa_config.question_nesting_level}"
            )
