from collections import OrderedDict
from functools import partial
from typing import Callable
from typing import Optional
from typing import Union

import dill  # type: ignore
import rich
from datasets import Dataset
from datasets import Split
from transformers import PreTrainedTokenizerFast

from .base_dm import BaseDataModule
from .corpus_dm import CorpusDataModule
from .datasets import medqa
from .pipelines.collate import CollateAsTensor
from .pipelines.collate import CollateTokens
from .pipelines.collate import MaybeSearchDocuments
from .pipelines.preprocessing import FormatAndTokenize
from .pipes import BlockSequential
from .pipes import Collate
from .pipes import DeCollate
from .pipes import Itemize
from .pipes import Parallel
from .pipes import RelevanceClassifier
from .pipes import Sequential
from .pipes import UpdateWith
from .pipes.tokenizer import CleanupPadTokens
from .utils.dataset import filter_questions_by_pos_docs
from .utils.dataset import get_column_names
from .utils.dataset import print_size_difference
from .utils.fingerprintable_map import FingerprintableMap
from .utils.transformations import set_example_idx
from .utils.typing import HgDataset
from fz_openqa.datamodules.pipelines.collate.nested_documents import (
    MaybeCollateDocuments,
)
from fz_openqa.datamodules.pipelines.collate.postprocess_docs import (
    PostprocessPipe,
)
from fz_openqa.tokenizers.static import ANS_TOKEN
from fz_openqa.tokenizers.static import QUERY_TOKEN
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pretty_decode


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
    ]

    # number of data points per subset train/val/test
    subset_size = [100, 20, 20]

    # number of options
    n_options = 4

    # Optional: corpus
    corpus: Optional[CorpusDataModule] = None

    # Optional: compiled dataset (pre-computed)
    compiled_dataset: Optional[HgDataset] = None

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
        compile_in_setup: bool = True,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.add_encoding_tokens = add_encoding_tokens

        # corpus object
        self.corpus = corpus

        # document retrieval
        self.compile_in_setup = compile_in_setup
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

        # the postprocessing pipe is applied after the documents are sampeld
        self.postprocessing = PostprocessPipe(
            self.relevance_classifier,
            n_retrieved_documents=self.n_retrieved_documents,
            n_select_documents=self.n_documents,
            max_select_pos_docs=self.max_pos_docs,
        )

        # define the collate operator
        self.collate_pipe: BlockSequential = self.get_collate_pipe()

        # compile
        if self.compile_in_setup:
            # todo: move this in a callback, so a model can be used
            self.build_index()
            self.compile_dataset(num_proc=self.num_proc)

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
            set_example_idx,
            batched=False,
            num_proc=self.num_proc,
            with_indices=True,
            desc="Indexing",
        )

        # cast to tensors
        # todo: check if we can remove this one totally
        # dataset.set_format(
        #     type="torch", columns=self.pt_attributes, output_all_columns=True
        # )
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

        # get a pipe to search document only if `n_retrieved_documents>0`
        # and documents are not already stored
        search_docs_pipe = MaybeSearchDocuments(
            self.n_retrieved_documents, corpus=self.corpus
        )

        # merge Q&A + documents (collate if available, else search)
        collate_qad = Sequential(
            Parallel(
                # A. pipe to collate documents if already stored (compiled dataset)
                MaybeCollateDocuments(self.tokenizer),
                # B. this one collate the question and answer fields
                base_qa_collate_pipe,
            ),
            # C. Query the corpus for documents, only if A. did not return any document
            UpdateWith(search_docs_pipe),
        )

        # get the post processing blocks (RelevanceClassifier + doc selection)
        postprocessing = (
            (k, UpdateWith(b)) for k, b in self.postprocessing.blocks.items()
        )

        return BlockSequential(
            [
                # A, B, C
                ("Collate Q&A + Search documents", collate_qad),
                # D: Classify documents +  F: select documents
                *postprocessing,
            ],
            id="collate-pipeline",
        )

    def get_qa_collate_pipe(self):
        # get the raw text questions, extract and collate
        raw_text_pipe = Collate(keys=["answer.text", "question.text"])
        # collate simple attributes
        simple_attr_pipe = CollateAsTensor(
            keys=["idx", "question.idx", "answer.target", "answer.n_options"]
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

    def compile_dataset(
        self,
        filter_unmatched: bool = True,
        num_proc: int = 1,
        batch_size: Optional[int] = 100,
    ):
        """
        Process the whole dataset with `collate_fn` and store
        into `self.compiled_dataset`

        NB: SystemExit: 15: is due to an error in huggingface dataset when attempting
        deleting the the dataset, see issue #114.
        """

        # unpack the Q&A collate pipe and the postprocessing
        qa_collate, *post = self.collate_pipe.blocks.items()

        # wrap p1 to avoid casting issues and avoid storing padding
        wrapped_p1 = Sequential(
            DeCollate(),
            qa_collate[1],
            Itemize(),
            CleanupPadTokens(self.tokenizer),
        )

        # store
        blocks = OrderedDict([(qa_collate[0], wrapped_p1), *post])

        # create a map operator on the fly for a given key and pipe
        def m(desc, block):
            # run dataset.map() with extra steps to allow safe caching and multiprocessing
            return FingerprintableMap(
                block,
                batched=True,
                num_proc=num_proc,
                batch_size=batch_size,
                desc=f"Compiling: {desc}",
            )

        dset = self.dataset
        for k, block in blocks.items():
            # block = Sequential(block, PrintBatch(f"out: {k}"))
            dset = m(k, block)(dset)
        self.compiled_dataset = dset

        # cast tensor values
        self.cast_compiled_dataset()

        if filter_unmatched:
            # filter out questions that are not match to any  positive document
            fn = partial(
                filter_questions_by_pos_docs, max_pos_docs=self.max_pos_docs
            )
            self.compiled_dataset = self.compiled_dataset.filter(fn)

        # print the difference in length for each split
        if self.verbose:
            print_size_difference(self.dataset, self.compiled_dataset)

    def build_index(self, model: Optional[Callable] = None, **kwargs):
        self.corpus.build_index(model=model, **kwargs)

    def cast_compiled_dataset(self):
        pt_cols = self.pt_attributes + self.corpus.pt_attributes
        pt_cols = [
            c for c in pt_cols if c in get_column_names(self.compiled_dataset)
        ]
        self.compiled_dataset.set_format(
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

    def get_dataset(self, split: Union[str, Split]) -> Dataset:
        """Return the dataset corresponding to the split,
        or the dataset itself if there is no split."""

        # use the compiled dataset if available
        dset = self.compiled_dataset or self.dataset

        # retrieve the dataset split
        return super().get_dataset(split, dset)
