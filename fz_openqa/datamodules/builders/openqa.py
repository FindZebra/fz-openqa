import json
import logging
from functools import partial
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import rich
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split

from fz_openqa.datamodules.builders.base import DatasetBuilder
from fz_openqa.datamodules.builders.corpus import CorpusBuilder
from fz_openqa.datamodules.builders.medqa import ConcatMedQaBuilder
from fz_openqa.datamodules.builders.medqa import MedQaBuilder
from fz_openqa.datamodules.index import FaissIndex
from fz_openqa.datamodules.index import Index
from fz_openqa.datamodules.index.builder import IndexBuilder
from fz_openqa.datamodules.index.pipes import FetchNestedDocuments
from fz_openqa.datamodules.index.pipes import SearchCorpus
from fz_openqa.datamodules.pipelines.collate.field import CollateField
from fz_openqa.datamodules.pipelines.preprocessing import FetchAndClassifyDocuments
from fz_openqa.datamodules.pipelines.preprocessing import SortDocuments
from fz_openqa.datamodules.pipes import BlockSequential
from fz_openqa.datamodules.pipes import Parallel
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import RelevanceClassifier
from fz_openqa.datamodules.pipes import SelectDocs
from fz_openqa.datamodules.utils.dataset import filter_questions_by_pos_docs
from fz_openqa.datamodules.utils.dataset import format_size_difference
from fz_openqa.datamodules.utils.dataset import get_column_names
from fz_openqa.datamodules.utils.map_with_fingerprint import MapWithFingerprint
from fz_openqa.datamodules.utils.typing import HfDataset
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pretty_decode

logger = logging.getLogger(__name__)


class OpenQaDataset(DatasetDict):
    def __init__(self, *, dataset: DatasetDict, corpus: Dataset, index: Index):
        super(OpenQaDataset, self).__init__(dataset)
        self.corpus = corpus
        self.index = index

    def new(self, *, dataset: DatasetDict) -> "OpenQaDataset":
        return OpenQaDataset(dataset=dataset, corpus=self.corpus, index=self.index)

    def __repr__(self):
        u = f"{self.__class__.__name__}:\n"
        u += f" - dataset={super().__repr__()}\n"
        u += f" - corpus={self.corpus}\n"
        u += f" - index={self.index}\n"
        return u


class OpenQaBuilder(DatasetBuilder):
    column_names = MedQaBuilder.column_names + [
        "document.row_idx",
        "document.retrieval_score",
        "document.match_score",
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
        num_proc: int = 2,
        batch_size: int = 100,
        **kwargs,
    ):
        super(OpenQaBuilder, self).__init__(cache_dir=None)

        # sub-builders
        self.dataset_builder = dataset_builder
        self.corpus_builder = corpus_builder

        # get the tokenizer
        self.tokenizer = dataset_builder.tokenizer
        assert self.tokenizer.vocab == corpus_builder.tokenizer.vocab

        # objects
        self.index_builder = index_builder

        # arguments
        self.n_documents = n_documents or n_retrieved_documents
        self.max_pos_docs = max_pos_docs
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
    def pt_attributes(self):
        return self.dataset_builder.pt_attributes + self.corpus_builder.pt_attributes

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

    def __call__(self, format: Optional[str] = "torch", **kwargs) -> OpenQaDataset:
        # de-activate formatting for the dataset to avoid messing up
        # with the newly added columns in map_dataset
        dataset = self.dataset_builder(format=None)
        corpus = self.corpus_builder()

        index = self.index_builder(dataset=corpus, **kwargs)

        dataset = self.map_dataset(dataset=dataset, corpus=corpus, index=index, **self.map_args)

        if format is not None:
            dataset = self.set_format(dataset, format=format)

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

        if isinstance(index, FaissIndex):
            collate_fn = CollateField(
                "question", tokenizer=self.tokenizer, exclude=["metamap", "text"], level=0
            )
            index.cache_query_dataset(dataset, collate_fn=collate_fn)

        # Search the document and tag them with `document.match_score`
        pipe = BlockSequential(
            [
                (
                    "Search documents",
                    SearchCorpus(
                        index,
                        k=n_retrieved_documents,
                        level=0,
                    ),
                ),
                (
                    "Classify documents",
                    FetchAndClassifyDocuments(
                        corpus_dataset=corpus,
                        classifier=relevance_classifier,
                        axis=1,
                        n=n_retrieved_documents,
                    ),
                ),
                ("Sort documents", SortDocuments()),
            ]
        )

        # process the dataset with each block
        original_size = {k: len(dset) for k, dset in dataset.items()}
        for k, block in pipe.blocks.items():
            logger.info(f"Processing: {k}")
            mapper = MapWithFingerprint(
                block,
                batched=True,
                cache_dir=self.dataset_builder.cache_dir,
                num_proc=num_proc,
                batch_size=batch_size,
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

    def get_collate_pipe(self) -> BlockSequential:
        """Build a Pipe to transform examples into a Batch."""

        # A. Collate all attributes stored in `self.dataset`
        collate_qad = Parallel(
            CollateField("document", tokenizer=self.tokenizer, level=1),
            self.dataset_builder.get_collate_pipe(),
        )

        # B. select documents (resample the field `document.row_idx`)
        select_documents = self.get_select_documents_pipe(
            self.n_documents,
            max_pos_docs=self.max_pos_docs,
        )

        # C. fetch documents attributes from `self.corpus` (e.g. document.input_ids, document.text)
        fetch_documents = FetchNestedDocuments(
            corpus_dataset=self.corpus_builder(),
            collate_pipe=self.corpus_builder.get_collate_pipe(),
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
    ) -> Optional[Pipe]:
        if n_documents == 0:
            return None

        return SelectDocs(
            total=n_documents,
            max_pos_docs=max_pos_docs,
            pos_select_mode="first",
            neg_select_mode="first",
            strict=False,
            update=True,
            level=level,
        )

    def format_row(self, row: Dict[str, Any]) -> str:
        decode_kwargs = {
            "skip_special_tokens": False,
            "tokenizer": self.tokenizer,
        }

        repr = self.dataset_builder.format_row(row)
        repr += get_separator("-") + "\n"
        repr += (
            f"* Documents: n={len(row['document.text'])}, "
            f"n_positive={sum(row['document.match_score'] > 0)}, "
            f"n_negative={sum(row['document.match_score'] == 0)}\n"
        )
        for j in range(min(len(row["document.text"]), 3)):
            repr += get_separator(".") + "\n"
            match_on = row.get("document.match_on", None)
            match_on = match_on[j] if match_on is not None else None
            repr += (
                f"|-* Document #{1 + j}, "
                f"score={row['document.retrieval_score'][j]:.2f}, "
                f"match_score={row['document.match_score'][j]}, "
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


class ConcatOpenQabuilder(OpenQaBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.dataset_builder, ConcatMedQaBuilder)

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

        if isinstance(index, FaissIndex):
            collate_fn = CollateField(
                "question", tokenizer=self.tokenizer, exclude=["metamap", "text"], level=0
            )
            index.cache_query_dataset(dataset, collate_fn=collate_fn)

        # Search the document and tag them with `document.match_score`
        pipe = BlockSequential(
            [
                (
                    "Search documents",
                    SearchCorpus(index, k=n_retrieved_documents, level=1),
                ),
                (
                    "Classify documents",
                    FetchAndClassifyDocuments(
                        corpus_dataset=corpus,
                        classifier=relevance_classifier,
                        axis=2,
                        n=n_retrieved_documents,
                        level=2,
                        extract_gold=False,
                    ),
                ),
                ("Sort documents", SortDocuments(level=2)),
            ]
        )

        # process the dataset with each block
        original_size = {k: len(dset) for k, dset in dataset.items()}
        for k, block in pipe.blocks.items():
            logger.info(f"Processing: {k}")
            mapper = MapWithFingerprint(
                block,
                batched=True,
                cache_dir=self.dataset_builder.cache_dir,
                num_proc=num_proc,
                batch_size=batch_size,
                desc=f"[Mapping] {k}",
            )
            dataset = mapper(dataset)

        # filter out questions that are not match to any  positive document
        if filter_unmatched:
            # todo: implement this
            raise NotImplementedError

            def fn(split: Split):
                return partial(
                    filter_questions_by_pos_docs,
                    n_documents=n_documents,
                    max_pos_docs=max_pos_docs,
                    split=split,
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

    def get_collate_pipe(self) -> BlockSequential:
        """Build a Pipe to transform examples into a Batch."""

        # A. Collate all attributes stored in `self.dataset`
        collate_qad = Parallel(
            CollateField("document", tokenizer=self.tokenizer, level=2),
            self.dataset_builder.get_collate_pipe(),
        )

        # B. select documents (resample the field `document.row_idx`)
        select_documents = self.get_select_documents_pipe(
            self.n_documents,
            max_pos_docs=self.max_pos_docs,
            level=2,
        )

        # C. fetch documents attributes from `self.corpus` (e.g. document.input_ids, document.text)
        fetch_documents = FetchNestedDocuments(
            corpus_dataset=self.corpus_builder(),
            collate_pipe=self.corpus_builder.get_collate_pipe(),
            level=2,
        )

        return BlockSequential(
            [
                ("Collate Q&A + document indexes", collate_qad),
                ("Select documents", select_documents),
                ("Fetch document data", fetch_documents),
            ],
            id="collate-pipeline",
        )

    def format_row(self, row: Dict[str, Any]) -> str:
        decode_kwargs = {
            "skip_special_tokens": False,
            "tokenizer": self.tokenizer,
        }

        repr = f"* Question #{row.get('question.idx', None)}\n"
        idx = row["answer.target"]

        # for each question-answer pair
        for i, an in enumerate(row["question.input_ids"]):
            locator = f"QA #{i+1}"
            repr += get_separator("-") + "\n"
            repr += f"|-* {locator}\n"
            # print question-answer pair
            an_style = "green" if idx == i else "cyan"
            line = (
                f"   - ({'x' if idx == i else ' '}) "
                f"{pretty_decode(an, **decode_kwargs, only_text=False, style=an_style)}\n"
            )
            repr += line

            # print documents attached to the question-answer pair
            repr += get_separator(".") + "\n"
            repr += f"|-* {locator} - Documents: n={len(row['document.text'][i])}"
            if "document.match_score" in row:
                repr += (
                    f", n_positive={sum(row['document.match_score'][i] > 0)}, "
                    f"n_negative={sum(row['document.match_score'][i] == 0)}"
                )
            repr += "\n"

            # for each document
            for j in range(min(len(row["document.text"][i]), 3)):
                match_on = row.get("document.match_on", None)
                match_on = match_on[i][j] if match_on is not None else None
                repr += f"|---* {locator} - Document #{1 + j}"
                if "document.match_score" in row:
                    repr += (
                        f", score={row['document.retrieval_score'][i][j]:.2f}, "
                        f"match_score={row['document.match_score'][i][j]}, "
                        f"match_on={match_on}"
                    )
                repr += "\n"

                doc_style = "yellow" if match_on else "white"
                repr += (
                    pretty_decode(
                        row["document.input_ids"][i][j],
                        **decode_kwargs,
                        style=doc_style,
                    )
                    + "\n"
                )

        return repr
