from functools import partial
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import rich
import torch
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split
from datasets.fingerprint import update_fingerprint
from transformers import PreTrainedTokenizerFast

from .base_dm import BaseDataModule
from .corpus_dm import CorpusDataModule
from .datasets import medqa
from .index import ElasticSearchIndex
from .pipes import AddPrefix
from .pipes import Apply
from .pipes import ApplyToAll
from .pipes import Collate
from .pipes import DeCollate
from .pipes import FilterKeys
from .pipes import Gate
from .pipes import Identity
from .pipes import Itemize
from .pipes import Lambda
from .pipes import Nest
from .pipes import Nested
from .pipes import Parallel
from .pipes import Pipe
from .pipes import RelevanceClassifier
from .pipes import Rename
from .pipes import ReplaceInKeys
from .pipes import SelectDocs
from .pipes import Sequential
from .pipes import Sort
from .pipes import TokenizerPipe
from .pipes.nesting import Flatten
from .pipes.nesting import flatten_nested
from .pipes.nesting import nested_list
from .utils import add_spec_token
from .utils import get_column_names
from .utils import HgDataset
from .utils import set_example_idx
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
        relevance_classifier: Optional[RelevanceClassifier] = None,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.add_encoding_tokens = add_encoding_tokens

        # corpus object
        if n_documents is not None:
            assert n_retrieved_documents > 0
        if n_retrieved_documents > 0:
            assert corpus is not None
        self.corpus = corpus
        self.n_retrieved_documents = n_retrieved_documents
        self.n_documents = n_documents or n_retrieved_documents
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
        super().setup(stage)

        if self.corpus is not None:
            self.corpus.setup()

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
            desc="Tokenizing questions and answers",
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

    def get_answer_tokenizer_pipe(self):
        """create a Pipe to tokenize the answer choices."""
        answer_text_pipes = Sequential(
            FilterKeys(lambda key: key == "answer.text"),
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
            FilterKeys(lambda key: "question.text" in key),
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
            ApplyToAll(op=lambda x: torch.tensor(x)),
        )

        # collate the questions attributes (question.input_ids, question.idx, ...)
        question_pipe = Sequential(
            Collate(keys=["question.input_ids", "question.attention_mask"]),
            ReplaceInKeys("question.", ""),
            Lambda(lambda batch: self.tokenizer.pad(batch)),
            AddPrefix("question."),
        )

        # collate answer options
        answer_pipe = Sequential(
            Collate(keys=["answer.input_ids", "answer.attention_mask"]),
            ReplaceInKeys("answer.", ""),
            Flatten(),
            Lambda(lambda batch: self.tokenizer.pad(batch)),
            Nest(stride=self.n_options),
            AddPrefix("answer."),
        )

        # process documents only if they are available
        documents_pipe = Gate(
            lambda exs: any("document." in k for eg in exs for k in eg.keys()),
            self.documents_collate_pipe(),
        )

        return Parallel(
            raw_text_pipe,
            simple_attr_pipe,
            question_pipe,
            answer_pipe,
            documents_pipe,
        )

    def documents_collate_pipe(self):
        """
        Build a pipe to collate a batch of retrieved document.
        """
        # get the raw text
        raw_text_pipe = FilterKeys(lambda key: key in ["document.text"])

        # Get the simple attribute and cast to tensor
        simple_attr_pipe = Sequential(
            FilterKeys(
                lambda key: key
                in [
                    "document.idx",
                    "document.passage_idx",
                    "document.retrieval_score",
                    "document.is_positive",
                ]
            ),
            ApplyToAll(op=lambda x: torch.tensor(x)),
        )

        # collate the questions attributes (question.input_ids, question.idx, ...)
        tokens_pipe = Sequential(
            FilterKeys(
                lambda key: key
                in ["document.input_ids", "document.attention_mask"]
            ),
            ReplaceInKeys("document.", ""),
            Lambda(self.tokenizer.pad),
            AddPrefix("document."),
        )

        return Sequential(
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

    def build_index(self, model: Optional[Callable] = None, **kwargs):
        self.corpus.build_index(model=model, **kwargs)

    def add_documents_to_batch(self, batch):
        """Query the corpus for a given batch of questions and collate."""
        corpus_batch = self.corpus.search_index(
            query=batch, k=self.n_retrieved_documents
        )
        corpus_batch = Rename({"idx": "document.global_idx"})(corpus_batch)
        batch.update(**corpus_batch)
        return self.postprocessing(batch)

    def get_postprocessing_pipe(
        self, relevance_classifier: RelevanceClassifier
    ) -> Pipe:
        if relevance_classifier is None:
            return Identity()

        # sort the documents based on score and `is_positive`
        sorter = Nested(
            Sequential(
                Sort(key="document.retrieval_score"),
                Sort(key="document.is_positive"),
            ),
            filter=lambda key: str(key).startswith("document."),
        )

        # select `n_documents`
        selector = SelectDocs(total=self.n_documents, max_pos_docs=1)

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
        or the dataset iteself if there is no split."""

        # if the dataset is compiled, return its split
        if self.compiled_dataset is not None:
            return super().get_dataset(split, self.compiled_dataset)

        # retrieve the dataset split
        return super().get_dataset(split)

    def collate_fn(self, examples: List[Batch]) -> Batch:
        """The function that is used to merge examples into a batch.
        Concatenating sequences with different length requires padding them."""
        batch = self.collate_pipe(examples)

        if self.compiled_dataset is None:
            if (
                self.n_retrieved_documents > 0
                and self.corpus.dataset is not None
            ):
                batch = self.add_documents_to_batch(batch)

        return batch

    def compile_dataset(self, filter_unmatched: bool = True):
        """Process the whole dataset with `collate_fn` and store
        into `self.compiled_dataset`"""

        # todo: clean padding
        pipe = Sequential(
            DeCollate(),
            self.collate_fn,
            Itemize(),
        )

        # Compile the dataset
        fingerprints = self.infer_compile_fingerprint()
        self.compiled_dataset = DatasetDict(
            {
                key: dset.map(
                    pipe,
                    batched=True,
                    num_proc=self.num_proc,
                    desc="Compiling dataset",
                    new_fingerprint=fingerprints.get(key, None),
                )
                for key, dset in self.dataset.items()
            }
        )

        # cast tensor values
        self.cast_compiled_dataset()

        if filter_unmatched:
            self.compiled_dataset = self.filter_unmatched_questions(
                self.compiled_dataset
            )

        # print the difference in length for each split
        if self.verbose:
            # store the previous split sizes
            prev_lengths = {k: len(v) for k, v in self.dataset.items()}
            new_lengths = {k: len(v) for k, v in self.compiled_dataset.items()}

            print(get_separator())
            rich.print(
                "> Dataset size after filtering questions with no positive document:"
            )
            for key in new_lengths.keys():
                rich.print(
                    f">  - {key}: {new_lengths[key]} ({100 * new_lengths[key] / prev_lengths[key]:.2f}%)"
                )
            print(get_separator())

    def filter_unmatched_questions(self, dataset: DatasetDict):
        def _filter(row):
            n = sum(row["document.is_positive"])
            return n > 0

        # filter the datasets
        return dataset.filter(_filter)

    def cast_compiled_dataset(self):
        pt_cols = self.pt_attributes + self.corpus.pt_attributes
        pt_cols = [
            c for c in pt_cols if c in get_column_names(self.compiled_dataset)
        ]
        self.compiled_dataset.set_format(
            type="torch", columns=pt_cols, output_all_columns=True
        )

    def infer_compile_fingerprint(self):
        if isinstance(self.corpus._index, ElasticSearchIndex):
            new_fingerprint = {
                k: update_fingerprint(
                    dset._fingerprint,
                    self.postprocessing,
                    {
                        "k": self.n_documents,
                        "es_index": self.corpus._index.index_name,
                    },
                )
                for k, dset in self.dataset.items()
            }
        else:
            new_fingerprint = {}
        return new_fingerprint
