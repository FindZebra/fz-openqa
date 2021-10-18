from functools import partial
from typing import Callable
from typing import Optional
from typing import Union

import dill  # type: ignore
import rich
import torch
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split
from transformers import PreTrainedTokenizerFast

from .base_dm import BaseDataModule
from .corpus_dm import CorpusDataModule
from .datasets import medqa
from .pipes import AddPrefix
from .pipes import Apply
from .pipes import ApplyToAll
from .pipes import Collate
from .pipes import DeCollate
from .pipes import FilterKeys
from .pipes import FirstEg
from .pipes import Gate
from .pipes import Identity
from .pipes import Itemize
from .pipes import Lambda
from .pipes import Nest
from .pipes import Nested
from .pipes import Parallel
from .pipes import Pipe
from .pipes import PrintBatch
from .pipes import PrintText
from .pipes import RelevanceClassifier
from .pipes import Rename
from .pipes import ReplaceInKeys
from .pipes import SearchCorpus
from .pipes import SelectDocs
from .pipes import Sequential
from .pipes import Sort
from .pipes import TokenizerPipe
from .pipes import Update
from .pipes.nesting import Flatten
from .pipes.nesting import flatten_nested
from .pipes.nesting import nested_list
from .pipes.tokenizer import CleanupPadTokens
from .utils.condition import HasKeyWithPrefix
from .utils.condition import Not
from .utils.condition import Reduce
from .utils.condition import Static
from .utils.dataset import filter_questions_by_pos_docs
from .utils.dataset import get_column_names
from .utils.dataset import print_size_difference
from .utils.filter_keys import KeyIn
from .utils.filter_keys import KeyWithPrefix
from .utils.fingerprintable_map import FingerprintableMap
from .utils.transformations import add_spec_token
from .utils.transformations import set_example_idx
from .utils.typing import HgDataset
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
    subset_size = [100, 10, 10]

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
        max_pos_docs: int = 1,
        relevance_classifier: Optional[RelevanceClassifier] = None,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.add_encoding_tokens = add_encoding_tokens

        # corpus object
        self.corpus = corpus

        # document retrieval
        if n_documents is not None:
            assert n_retrieved_documents > 0
            assert n_documents <= n_retrieved_documents

        if n_retrieved_documents > 0:
            assert corpus is not None

        self.n_retrieved_documents = n_retrieved_documents
        self.n_documents = n_documents or n_retrieved_documents
        self.max_pos_docs = max_pos_docs
        # the postprocessing pipe is applied after the documents are sampeld
        self.postprocessing = self.get_postprocessing_pipe(
            relevance_classifier
        )

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
            assert self.n_documents <= len(self.corpus.dataset), (
                f"The corpus is too small to retrieve that many documents\n:"
                f"n_documents={self.n_documents} > n_passages={len(self.corpus.dataset)}"
            )

        # define the collate operator
        self.collate_pipe = self.get_collate_pipe()

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
        dataset.set_format(
            type="torch", columns=self.pt_attributes, output_all_columns=True
        )
        return dataset

    def build_index(self, model: Optional[Callable] = None, **kwargs):
        self.corpus.build_index(model=model, **kwargs)

    def get_answer_tokenizer_pipe(self):
        """create a Pipe to tokenize the answer choices."""
        answer_text_pipes = Sequential(
            FilterKeys(KeyIn(["answer.text"])),
            ReplaceInKeys("answer.", ""),
            ApplyToAll(flatten_nested, element_wise=False),
            Apply(
                {"text": partial(add_spec_token, ANS_TOKEN)}, element_wise=True
            )
            if self.add_encoding_tokens
            else None,
            TokenizerPipe(
                self.tokenizer,
                max_length=self.max_length,
                fields="text",
                return_token_type_ids=False,
                add_special_tokens=False,
                return_offsets_mapping=False,
            ),
            ApplyToAll(partial(nested_list, stride=self.n_options)),
            AddPrefix("answer."),
        )
        return answer_text_pipes

    def get_question_tokenizer_pipe(self):
        """create a Pipe to tokenize the questions."""
        question_pipes = Sequential(
            FilterKeys(KeyIn(["question.text"])),
            ReplaceInKeys("question.", ""),
            Apply(
                {"text": partial(add_spec_token, QUERY_TOKEN)},
                element_wise=True,
            )
            if self.add_encoding_tokens
            else None,
            TokenizerPipe(
                self.tokenizer,
                max_length=self.max_length,
                fields="text",
                return_token_type_ids=False,
                add_special_tokens=False,
                return_offsets_mapping=False,
            ),
            AddPrefix("question."),
        )
        return question_pipes

    def get_collate_pipe(self) -> Pipe:
        """Build a Pipe to transform examples into a Batch."""

        # get the raw text questions, extract and collate
        raw_text_pipe = Collate(keys=["answer.text", "question.text"])

        # collate simple attributes
        simple_attr_pipe = Sequential(
            Collate(keys=["idx", "answer.target", "answer.n_options"]),
            ApplyToAll(torch.tensor),
            id="simple-attrs",
        )

        # collate the questions attributes (question.input_ids, question.idx, ...)
        question_pipe = Sequential(
            Collate(keys=["question.input_ids", "question.attention_mask"]),
            ReplaceInKeys("question.", ""),
            Lambda(self.tokenizer.pad),
            AddPrefix("question."),
            id="questions",
        )

        # collate answer options
        answer_pipe = Sequential(
            Collate(keys=["answer.input_ids", "answer.attention_mask"]),
            ReplaceInKeys("answer.", ""),
            Flatten(),
            Lambda(self.tokenizer.pad),
            Nest(stride=self.n_options),
            AddPrefix("answer."),
            id="answers",
        )

        # the full pipe to collate question and answer fields
        qa_collate_pipe = Parallel(
            raw_text_pipe,
            simple_attr_pipe,
            question_pipe,
            answer_pipe,
            id="qa-collate",
        )

        # Search corpus pipe: this pipe is activated only if
        # the batch does not already contains documents (Gate).
        condition = Reduce(
            Static(self.n_retrieved_documents > 0),
            Not(HasKeyWithPrefix("document.")),
            reduce_op=all,
        )
        # todo: do not initialize SearchCorpus when no corpus is available
        search_docs_pipe = Gate(
            condition,
            Sequential(
                Update(
                    Sequential(
                        SearchCorpus(
                            self.corpus, k=self.n_retrieved_documents
                        ),
                        Rename({"idx": "document.global_idx"}),
                    )
                ),
                self.postprocessing,
            ),
        )

        return Sequential(
            Parallel(
                # A. pipe to collate documents if already stored (compiled dataset)
                self.get_documents_collate_pipe(),
                # B. this one collate the question and answer fields
                qa_collate_pipe,
            ),
            # C. Query the corpus for documents, only if A. did not return anything
            Update(search_docs_pipe),
            id="collate-fn",
        )

    def get_documents_collate_pipe(self):
        """
        Build a pipe to collate a batch of retrieved document.
        This is used only if documents are already stored with each q-a pair,
        which is the case when compiling datasets.
        """
        # get the raw text
        raw_text_pipe = FilterKeys(KeyIn(["document.text"]))

        # Get the simple attribute and cast to tensor
        simple_attr_pipe = Sequential(
            FilterKeys(
                KeyIn(
                    [
                        "document.idx",
                        "document.passage_idx",
                        "document.retrieval_score",
                        "document.is_positive",
                    ]
                )
            ),
            ApplyToAll(op=torch.tensor),
        )

        # collate the questions attributes (question.input_ids, question.idx, ...)
        tokens_pipe = Sequential(
            FilterKeys(
                KeyIn(["document.input_ids", "document.attention_mask"])
            ),
            ReplaceInKeys("document.", ""),
            Lambda(self.tokenizer.pad),
            AddPrefix("document."),
        )

        # the full pipe used to collate documents
        doc_collate_pipe = Sequential(
            Collate(
                keys=[
                    "document.idx",
                    "document.passage_idx",
                    "document.retrieval_score",
                    "document.is_positive",
                    "document.positive_count",
                    "document.input_ids",
                    "document.attention_mask",
                    "document.text",
                ]
            ),
            Flatten(),
            Parallel(raw_text_pipe, simple_attr_pipe, tokens_pipe),
            Nest(stride=self.n_documents),
        )

        condition = Sequential(FirstEg(), HasKeyWithPrefix("document."))
        return Gate(condition, doc_collate_pipe)

    def get_postprocessing_pipe(
        self, relevance_classifier: RelevanceClassifier
    ) -> Pipe:
        """Get the pipe that classify documents as `is_positive`,
        sort them and select `n_documents` among the `n_retrieved_docs`"""
        if relevance_classifier is None:
            return Identity()

        # sort the documents based on score and `is_positive`
        sorter = Nested(
            Sequential(
                Sort(key="document.retrieval_score"),
                Sort(key="document.is_positive"),
            ),
            filter=KeyWithPrefix("document."),
        )

        # select `n_documents` where count(is_positive) <= max_pos_docs
        selector = SelectDocs(
            total=self.n_documents,
            max_pos_docs=self.max_pos_docs,
            strict=False,
        )

        return Sequential(
            relevance_classifier,
            sorter,
            selector,
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

    def compile_dataset(
        self,
        filter_unmatched: bool = True,
        num_proc: int = 1,
        batch_size: int = 100,
    ):
        """Process the whole dataset with `collate_fn` and store
        into `self.compiled_dataset`"""

        pipe = Sequential(
            DeCollate(),
            self.collate_pipe,
            Itemize(),
            CleanupPadTokens(self.tokenizer),
        )

        # run dataset.map() with extra steps to allow safe caching and multiprocessing
        mapper = FingerprintableMap(
            pipe,
            batched=True,
            num_proc=num_proc,
            batch_size=batch_size,
            desc="Compiling dataset",
        )

        self.compiled_dataset = mapper(self.dataset)

        # cast tensor values
        self.cast_compiled_dataset()

        if filter_unmatched:
            # filter out questions that are not match to any  positive document
            self.compiled_dataset = self.filter_unmatched_questions(
                self.compiled_dataset
            )

        # print the difference in length for each split
        if self.verbose:
            print_size_difference(self.dataset, self.compiled_dataset)

    def filter_unmatched_questions(self, dataset: DatasetDict):
        fn = partial(
            filter_questions_by_pos_docs, max_pos_docs=self.max_pos_docs
        )
        return dataset.filter(fn)

    def cast_compiled_dataset(self):
        pt_cols = self.pt_attributes + self.corpus.pt_attributes
        pt_cols = [
            c for c in pt_cols if c in get_column_names(self.compiled_dataset)
        ]
        self.compiled_dataset.set_format(
            type="torch", columns=pt_cols, output_all_columns=True
        )
