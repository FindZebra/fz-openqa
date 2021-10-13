from functools import partial
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import rich
import torch
from datasets import Dataset
from datasets import Split
from transformers import PreTrainedTokenizerFast

from .base_dm import BaseDataModule
from .corpus_dm import CorpusDataModule
from .datasets import medqa
from .pipes import AddPrefix
from .pipes import Apply
from .pipes import ApplyToAll
from .pipes import Collate
from .pipes import FilterKeys
from .pipes import Gate
from .pipes import Lambda
from .pipes import Nest
from .pipes import Parallel
from .pipes import Pipe
from .pipes import RelevanceClassifier
from .pipes import ReplaceInKeys
from .pipes import Sequential
from .pipes import TokenizerPipe
from .sampler.corpus_sampler import CorpusSampler
from .utils import add_spec_token
from .utils import flatten_nested
from .utils import HgDataset
from .utils import nested_list
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

    # optional corpus
    corpus: Optional[CorpusDataModule] = None

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        add_encoding_tokens: bool = True,
        corpus: Optional[BaseDataModule] = None,
        n_documents: int = 0,
        relevance_classifier: Optional[RelevanceClassifier] = None,
        use_corpus_sampler: bool = False,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.add_encoding_tokens = add_encoding_tokens

        # corpus object
        if n_documents > 0:
            assert corpus is not None
        self.corpus = corpus
        self.n_documents = n_documents
        self.relevance_classifier = relevance_classifier
        # sample the corpus, before or after the collate function
        # when using the corpus sampler, documents are fetched for each question in __getitem__
        # without corpus sampler, documents are fetched in the collate_fn (in batch).
        self.use_corpus_sampler = use_corpus_sampler

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
            assert self.n_documents <= len(self.corpus.dataset), (
                f"The corpus is too small to retrieve that many documents\n:"
                f"n_documents={self.n_documents} > n_passages={len(self.corpus.dataset)}"
            )

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
            ApplyToAll(partial(nested_list, stride=4)),
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
        raw_text_pipe = Collate(keys=["answer.text", "question.text", "answer.cui", "answer.synonyms"])

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
            ApplyToAll(flatten_nested, element_wise=False),
            Lambda(lambda batch: self.tokenizer.pad(batch)),
            ApplyToAll(lambda x: x.view(-1, self.n_options, x.size(-1))),
            AddPrefix("answer."),
        )

        # Optional: get the pipe to process the sampled documents
        documents_pipe = Gate(
            self.use_corpus_sampler and self.n_documents > 0,
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
            ApplyToAll(flatten_nested, element_wise=False),
            Parallel(raw_text_pipe, simple_attr_pipe, tokens_pipe),
            Nest(stride=self.n_documents),
        )

    def build_index(self, model: Optional[Callable] = None, **kwargs):
        self.corpus.build_index(model=model, **kwargs)

    def add_documents_to_batch(self, batch):
        """Query the corpus for a given batch of questions and collate."""
        corpus_batch = self.corpus.search_index(
            query=batch, k=self.n_documents
        )
        batch.update(**corpus_batch)

        if self.relevance_classifier is not None:
            batch = self.relevance_classifier(batch)
        return batch

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
        dataset = super().get_dataset(split)
        if self.use_corpus_sampler:
            if self.n_documents > 0 and self.corpus.dataset is not None:
                dataset = CorpusSampler(
                    dataset,
                    corpus=self.corpus,
                    n_documents=self.n_documents,
                    relevance_classifier=self.relevance_classifier,
                )

        return dataset

    def collate_fn(self, examples: List[Batch]) -> Batch:
        """The function that is used to merge examples into a batch.
        Concatenating sequences with different length requires padding them."""
        batch = self.collate_pipe(examples)

        if not self.use_corpus_sampler:
            if self.n_documents > 0 and self.corpus.dataset is not None:
                batch = self.add_documents_to_batch(batch)

        return batch
