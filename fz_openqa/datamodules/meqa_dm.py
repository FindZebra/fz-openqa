import logging
from functools import partial
from typing import Callable
from typing import Optional
from typing import Union

import dill  # type: ignore
import rich
from datasets import Dataset
from datasets import Split
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerFast

from .base_dm import BaseDataModule
from .corpus_dm import CorpusDataModule
from .datasets import medqa
from .pipelines.collate import CollateAsTensor
from .pipelines.collate import CollateTokens
from .pipelines.preprocessing import ClassifyDocuments
from .pipelines.preprocessing import FormatAndTokenize
from .pipelines.preprocessing import SearchDocuments
from .pipelines.preprocessing import SortDocuments
from .pipes import AsBatch
from .pipes import AsFlatten
from .pipes import BlockSequential
from .pipes import Collate
from .pipes import FilterKeys
from .pipes import Parallel
from .pipes import Pipe
from .pipes import RelevanceClassifier
from .pipes import SelectDocs
from .pipes import Sequential
from .pipes import UpdateWith
from .pipes.search import FeatchDocuments
from .utils.dataset import filter_questions_by_pos_docs
from .utils.dataset import get_column_names
from .utils.dataset import print_size_difference
from .utils.fingerprintable_map import FingerprintableMap
from .utils.transformations import set_row_idx
from .utils.typing import HgDataset
from fz_openqa.datamodules.pipelines.collate.nested_documents import (
    MaybeCollateDocuments,
)
from fz_openqa.tokenizers.static import ANS_TOKEN
from fz_openqa.tokenizers.static import QUERY_TOKEN
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pretty_decode

logger = logging.getLogger(__name__)


class MedQaDataModule(BaseDataModule):
    """A base DataModule for question answering."""

    # HuggingFace dataset id or local path to script
    dset_script_path_or_id = medqa.__file__

    # name of the attributes that will be converted to
    # tensors in the preprocessing function
    pt_attributes = [
        "question.input_ids",
        "question.attention_mask",
        "question.idx",
        "answer.input_ids",
        "answer.attention_mask",
        "answer.target",
        "document.match_score",
        "document.retrieval_score",
    ]

    # number of data points per subset train/val/test
    subset_size = [100, 50, 50]

    # number of options
    n_options = 4

    # Optional: corpus
    corpus: Optional[CorpusDataModule] = None

    # check if the dataset is mapped with the corpus
    _is_mapped = False

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        add_encoding_tokens: bool = True,
        corpus: Optional[BaseDataModule] = None,
        n_retrieved_documents: int = 0,
        n_documents: Optional[int] = None,
        max_pos_docs: Optional[int] = 1,
        relevance_classifier: Optional[RelevanceClassifier] = None,
        map_corpus_batch_size: int = 100,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.add_encoding_tokens = add_encoding_tokens

        # corpus object
        self.corpus = corpus

        # document retrieval
        self.map_corpus_batch_size = map_corpus_batch_size
        if n_documents is not None:
            assert n_retrieved_documents > 0
            assert n_documents <= n_retrieved_documents

        if n_retrieved_documents > 0:
            assert corpus is not None

        self.n_retrieved_documents = n_retrieved_documents
        self.n_documents = n_documents or n_retrieved_documents
        self.max_pos_docs = max_pos_docs
        self.relevance_classifier = relevance_classifier

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        super().prepare_data()
        if self.corpus is not None:
            self.corpus.prepare_data()

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        # load the dataset and potentially filter it
        self.dataset = self.load_and_filter_dataset()

        # preprocess
        self.dataset = self.preprocess_dataset(self.dataset)

        # setup the corpus object
        if self.corpus is not None:
            self.corpus.setup()
            assert self.corpus._index is not None, "An Index must be provided."
            assert self.n_documents <= len(self.corpus.dataset), (
                f"The corpus is too small to retrieve that many documents\n:"
                f"n_documents={self.n_documents} > n_passages={len(self.corpus.dataset)}"
            )

        # define the collate operator
        self.collate_pipe: BlockSequential = self.get_collate_pipe()

        # map the questions with documents from the corpus
        if self.n_retrieved_documents > 0 and not self._is_mapped:
            assert self.corpus is not None, "A corpus must be set and set up."
            self.build_index()
            self.map_corpus(
                num_proc=self.num_proc,
                verbose=self.verbose,
                batch_size=self.map_corpus_batch_size,
            )

            # cast features as tensors
            self.cast_dataset_as_tensors(self.dataset)

    def preprocess_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply processing steps to the dataset.
        Tokenization and formatting as PyTorch tensors"""

        # Tokenize the text fields (question and answers)
        dataset = dataset.map(
            Parallel(
                self.get_question_tokenizer_pipe(),
                self.get_answer_tokenizer_pipe(),
            ),
            batched=True,
            num_proc=self.num_proc,
            desc="Tokenizing documents and extracting overlapping passages",
        )

        # add an index column
        dataset = dataset.map(
            partial(set_row_idx, key="question.row_idx"),
            batched=False,
            num_proc=self.num_proc,
            with_indices=True,
            desc="Indexing",
        )

        return dataset

    def get_answer_tokenizer_pipe(self):
        return FormatAndTokenize(
            prefix="answer.",
            text_formatter=self.text_formatter,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            add_encoding_tokens=self.add_encoding_tokens,
            spec_tokens=ANS_TOKEN,
            stride=self.n_options,
        )

    def get_question_tokenizer_pipe(self):
        """create a Pipe to tokenize the questions."""
        return FormatAndTokenize(
            prefix="question.",
            text_formatter=self.text_formatter,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            add_encoding_tokens=self.add_encoding_tokens,
            spec_tokens=QUERY_TOKEN,
            stride=None,
        )

    def get_collate_pipe(self) -> BlockSequential:
        """Build a Pipe to transform examples into a Batch."""

        # collate questions and couments
        base_qa_collate_pipe = self.get_qa_collate_pipe()

        # merge Q&A + documents (collate if available, else search)
        collate_qad = Sequential(
            Parallel(
                # A. pipe to collate documents if already stored (compiled dataset)
                MaybeCollateDocuments(self.tokenizer),
                # B. this one collate the question and answer fields
                base_qa_collate_pipe,
            ),
        )

        # C. select documents
        select_documents = self.get_select_documents_pipe(
            self.n_documents or self.n_retrieved_documents,
            max_pos_docs=self.max_pos_docs,
        )

        # D. fetch documents attributes (input_ids)
        fetch_documents = None  # self.get_fetch_documents_pipe(self.corpus)

        return BlockSequential(
            [
                # A, B: collate QA fields
                ("Collate Q&A + document indexes", collate_qad),
                # C: select documents
                ("Select documents", select_documents),
                # D: Fetch all document fields
                ("Fetch document data", fetch_documents),
            ],
            id="collate-pipeline",
        )

    @staticmethod
    def get_select_documents_pipe(
        n_documents: int, *, max_pos_docs: Optional[int]
    ) -> Optional[Pipe]:
        if n_documents == 0:
            return None

        return SelectDocs(
            total=n_documents,
            max_pos_docs=max_pos_docs,
            pos_select_mode="first",
            neg_select_mode="first",
            strict=False,
        )

    def get_fetch_documents_pipe(
        self, corpus: Optional[CorpusDataModule]
    ) -> Optional[Pipe]:
        if corpus is None:
            return None

        fetch_documents = Sequential(
            FilterKeys(lambda key: key in ["document.row_idx"]),
            AsFlatten(
                FeatchDocuments(
                    dataset=corpus.dataset,
                    collate_pipe=corpus.collate_pipe,
                )
            ),
        )
        return UpdateWith(fetch_documents)

    def get_qa_collate_pipe(self):
        # get the raw text questions, extract and collate
        raw_text_pipe = Collate(
            keys=[
                "answer.text",
                "question.text",
                "answer.synonyms",
                "answer.cui",
                "question.metamap",
            ]
        )
        # collate simple attributes
        simple_attr_pipe = CollateAsTensor(
            keys=[
                "question.row_idx",
                "question.idx",
                "answer.target",
                "answer.n_options",
            ]
        )
        # collate the questions attributes (question.input_ids, question.idx, ...)
        question_pipe = CollateTokens("question.", tokenizer=self.tokenizer)
        # collate answer options
        answer_pipe = CollateTokens(
            "answer.", tokenizer=self.tokenizer, stride=self.n_options
        )
        # the full pipe to collate question and answer fields
        base_qa_collate_pipe = Parallel(
            raw_text_pipe,
            simple_attr_pipe,
            question_pipe,
            answer_pipe,
            id="base-qa-collate",
        )
        return base_qa_collate_pipe

    def map_corpus(
        self,
        filter_unmatched: bool = True,
        num_proc: int = 1,
        batch_size: Optional[int] = 10,
        verbose: bool = False,
    ):
        """
        Map the dataset with documents from the corpus.

        NB: SystemExit: 15: is due to an error in huggingface dataset when attempting
        deleting the the dataset, see issue #114.
        """

        # Search the document and tag them with `document.match_score`
        pipe = BlockSequential(
            [
                (
                    "Search documents",
                    SearchDocuments(
                        corpus=self.corpus,
                        n_documents=self.n_retrieved_documents,
                    ),
                ),
                (
                    "Classify documents",
                    ClassifyDocuments(
                        dataset=self.corpus.dataset,
                        relevance_classifier=self.relevance_classifier,
                    ),
                ),
                ("Sort documents", SortDocuments()),
            ]
        )

        def get_cached_pipe(desc, block):
            """"run dataset.map() with extra steps to allow safe caching and multiprocessing"""
            return FingerprintableMap(
                block,
                batched=True,
                num_proc=num_proc,
                batch_size=batch_size,
                desc=f"Map questions: {desc}",
                verbose=verbose,
            )

        # process the dataset with each block
        original_size = {k: len(dset) for k, dset in self.dataset.items()}
        for k, block in pipe.blocks.items():
            logger.info(f"Processing: {k}")
            pipe_k = get_cached_pipe(k, block)
            self.dataset = pipe_k(self.dataset)

        # filter out questions that are not match to any  positive document
        if filter_unmatched:
            fn = partial(
                filter_questions_by_pos_docs, max_pos_docs=self.max_pos_docs
            )
            self.dataset = self.dataset.filter(fn)

        # print the difference in length for each split
        if verbose:
            print_size_difference(original_size, self.dataset)

        self._is_mapped = True

    def build_index(self, model: Optional[Callable] = None, **kwargs):
        self.corpus.build_index(model=model, **kwargs)

    def cast_dataset_as_tensors(self, dataset: Dataset):
        """Cast dataset features as tensors"""
        pt_cols = self.pt_attributes
        if self.corpus is not None:
            pt_cols += self.corpus.pt_attributes

        # filter columns
        pt_cols = [c for c in pt_cols if c in get_column_names(dataset)]

        # set the format
        dataset.set_format(
            type="torch", columns=pt_cols, output_all_columns=True
        )

    def display_one_sample(self, example: Batch):
        """Decode and print one example from the batch"""
        decode_kwargs = {
            "skip_special_tokens": False,
            "tokenizer": self.tokenizer,
        }
        print("* Question:")
        rich.print(
            pretty_decode(example["question.input_ids"], **decode_kwargs)
        )

        print(get_separator())
        print("* Answer Choices:")
        idx = example["answer.target"]
        for i, an in enumerate(example["answer.input_ids"]):
            print(
                f"   - [{'x' if idx == i else ' '}] "
                f"{self.tokenizer.decode(an, **decode_kwargs).replace('[PAD]', '').strip()}"
            )

    def get_dataset(
        self, split: Union[str, Split], dataset: Optional[HgDataset] = None
    ) -> Union[TorchDataset, Dataset]:
        """Return the dataset corresponding to the split,

        or the dataset itself if there is no split."""

        class FetchDocuments(TorchDataset):
            def __init__(self, dataset: Dataset, fetch_document_pipe: Pipe):
                self.dataset = dataset
                self.pipe = UpdateWith(AsBatch(fetch_document_pipe))

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, item):
                x = self.dataset[item]
                return self.pipe(x)

        # retrieve the dataset split
        dataset = super().get_dataset(split, dataset or self.dataset)
        return FetchDocuments(
            dataset, self.get_fetch_documents_pipe(self.corpus)
        )
