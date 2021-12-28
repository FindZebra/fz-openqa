import logging
from functools import partial
from typing import Any
from typing import Dict
from typing import Optional

import dill  # type: ignore
from datasets import DatasetDict
from datasets import load_dataset

from ...utils.datastruct import Eg
from ..pipelines.collate.field import CollateField
from ..pipes.answer_options import ConcatTextFields
from ..pipes.control.condition import In
from ..pipes.nesting import ApplyAsFlatten
from ..pipes.nesting import Expand
from ..utils.dataset import format_size_difference
from .hf_dataset import HfDatasetBuilder
from fz_openqa.datamodules.generators import medqa_us_custom
from fz_openqa.datamodules.pipelines.preprocessing import FormatAndTokenize
from fz_openqa.datamodules.pipes import Apply
from fz_openqa.datamodules.pipes import Parallel
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.utils.transformations import add_spec_token
from fz_openqa.datamodules.utils.transformations import set_row_idx
from fz_openqa.datamodules.utils.typing import HfDataset
from fz_openqa.tokenizers.static import ANS_TOKEN
from fz_openqa.tokenizers.static import QUERY_TOKEN
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pretty_decode

logger = logging.getLogger(__name__)


class MinLength:
    def __init__(self, key: str, min_length: int):
        self.key = key
        self.min_length = min_length

    def __call__(self, row: Eg, **kwargs) -> bool:
        x = row[self.key]
        if isinstance(x, str):
            return len(x) >= self.min_length
        elif isinstance(x, list):
            return all(len(y) >= self.min_length for y in x)
        else:
            raise TypeError(f"{self.key} is not a string or list")


class MedQaBuilder(HfDatasetBuilder):
    # HuggingFace dataset id or local path to script
    dset_script_path_or_id = medqa_us_custom.__file__

    # nesting level of the question field
    nesting_level = 0

    # name of the attributes that will be converted to
    # tensors in the preprocessing function
    pt_attributes = [
        "question.input_ids",
        "question.attention_mask",
        "question.idx",
        "question.row_idx",
        "answer.input_ids",
        "answer.attention_mask",
        "answer.target",
        "document.match_score",
        "document.retrieval_score",
    ]

    # number of data points per subset train/val/test
    subset_size = [1000, 100, 100]

    # number of options
    n_options = 4

    # output columns
    column_names = [
        "answer.text",
        "answer.input_ids",
        "answer.attention_mask",
        "answer.target",
        "question.text",
        "question.input_ids",
        "question.attention_mask",
    ]

    def __init__(
        self, *args, min_answer_length: Optional[int] = None, n_query_tokens: int = 1, **kwargs
    ):
        super(MedQaBuilder, self).__init__(*args, **kwargs)
        self.min_answer_length = min_answer_length
        self.n_query_tokens = n_query_tokens

    def filter_dataset(self, dataset: HfDataset) -> HfDataset:
        """Apply filter operation to the dataset and return"""
        return dataset

    def preprocess_dataset(self, dataset: HfDataset) -> HfDataset:
        """Apply processing steps to the dataset.
        Tokenization and formatting as PyTorch tensors"""

        # filter out answers that are too short
        if self.min_answer_length is not None:
            logger.info(f"Filtering out answers shorter than {self.min_answer_length} characters")
            lengths = {k: len(d) for k, d in dataset.items()}
            dataset = dataset.filter(MinLength("answer.text", self.min_answer_length))
            logger.info(format_size_difference(lengths, dataset))

        # Tokenize the text fields (question and answers)
        if self.tokenizer:
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
            partial(set_row_idx, key="question.row_idx"),
            batched=True,
            batch_size=1000,
            num_proc=self.num_proc,
            with_indices=True,
            desc="Indexing rows",
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
            shape=[-1, self.n_options],
        )

    def get_question_tokenizer_pipe(self):
        """create a Pipe to tokenize the questions."""
        return FormatAndTokenize(
            prefix="question.",
            text_formatter=self.text_formatter,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            add_encoding_tokens=self.add_encoding_tokens,
            add_special_tokens=self.add_special_tokens,
            spec_tokens=self.n_query_tokens * [QUERY_TOKEN],
            shape=None,
        )

    def _get_collate_pipe(self):
        # get the raw text questions, extract and collate
        return Parallel(
            CollateField("question", tokenizer=self.tokenizer, level=0, id="collate-questions"),
            CollateField(
                "answer",
                level=0,
                exclude=["input_ids", "attention_mask"],
                to_tensor=["target"],
                id="collate-answer-attributes",
            ),
            CollateField(
                "answer",
                tokenizer=self.tokenizer,
                level=1,
                include_only=["input_ids", "attention_mask"],
                id="pad-answer-tokens",
            ),
        )

    def format_row(self, row: Dict[str, Any]) -> str:
        """Decode and print one row from the batch"""
        decode_kwargs = {
            "skip_special_tokens": False,
            "tokenizer": self.tokenizer,
        }
        u = "* Question:"
        u += (
            pretty_decode(
                row["question.input_ids"],
                **decode_kwargs,
                style="deep_sky_blue3",
            )
            + "\n"
        )

        u += get_separator("-") + "\n"
        u += "* Answer Choices:" + "\n"
        idx = row["answer.target"]
        for i, an in enumerate(row["answer.input_ids"]):
            an_style = "green" if idx == i else "white"
            line = (
                f"   - ({'x' if idx == i else ' '}) "
                f"{pretty_decode(an, **decode_kwargs, only_text=True, style=an_style)}\n"
            )
            u += line

        return u


class ConcatMedQaBuilder(MedQaBuilder):
    """A MedQa dataset with concatenated questions and answers"""

    # nesting level of the question field
    nesting_level = 1

    # name of the attributes that will be converted to
    # tensors in the preprocessing function
    pt_attributes = [
        "question.input_ids",
        "question.attention_mask",
        "answer.target",
        "document.match_score",
        "document.retrieval_score",
    ]

    # output columns
    column_names = [
        "question.text",
        "question.input_ids",
        "question.attention_mask",
        "answer.text",
        "answer.target",
    ]

    def preprocess_dataset(self, dataset: HfDataset) -> HfDataset:
        """Apply processing steps to the dataset.
        Tokenization and formatting as PyTorch tensors"""

        # concat question and answers
        dataset = dataset.map(
            self.get_concat_qa_pipe(),
            batched=True,
            num_proc=self.num_proc,
            desc="Concatenating questions and answers",
        )

        # Tokenize the text fields (question and answers)
        dataset = dataset.map(
            self.get_qa_tokenizer_pipe(),
            batched=True,
            num_proc=self.num_proc,
            desc="Tokenizing questions and answers",
        )

        # add an index column
        dataset = dataset.map(
            partial(set_row_idx, key="question.row_idx"),
            batched=True,
            num_proc=self.num_proc,
            with_indices=True,
            desc="Indexing",
        )

        return dataset

    def get_concat_qa_pipe(self):
        q_start_tokens = []
        if self.add_special_tokens:
            q_start_tokens.append(self.tokenizer.sep_token)
        if self.add_encoding_tokens:
            q_start_tokens.extend(self.n_query_tokens * [QUERY_TOKEN])

        add_spec_tokens_pipe = Apply(
            {"question.text": partial(add_spec_token, q_start_tokens)}, element_wise=True
        )

        return Sequential(
            self.text_formatter.copy(
                text_key=["question.text", "answer.text"]
            ),  # <- added this line here
            add_spec_tokens_pipe,
            Expand(axis=1, n=self.n_options, update=True, input_filter=In(["question.text"])),
            ApplyAsFlatten(
                ConcatTextFields(keys=["answer.text", "question.text"], new_key="question.text"),
                level=1,
            ),
            input_filter=In(["question.text", "answer.text"]),
        )

    def get_qa_tokenizer_pipe(self):
        return FormatAndTokenize(
            prefix="question.",
            text_formatter=None,  # <- changed here
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            add_encoding_tokens=self.add_encoding_tokens,
            add_special_tokens=self.add_special_tokens,
            spec_tokens=ANS_TOKEN,
            shape=[-1, self.n_options],
        )

    def _get_collate_pipe(self):
        # get the raw text questions, extract and collate
        return Parallel(
            CollateField(
                "question", exclude=["input_ids", "attention_mask"], level=0, id="collate-questions"
            ),
            CollateField(
                "answer",
                level=0,
                exclude=["input_ids", "attention_mask"],
                to_tensor=["target"],
                id="collate-answer-attributes",
            ),
            CollateField(
                "question",
                tokenizer=self.tokenizer,
                level=1,
                include_only=["input_ids", "attention_mask"],
                id="pad-nested-question-tokens",
            ),
        )

    def format_row(self, row: Dict[str, Any]) -> str:
        """Decode and print one row from the batch"""
        decode_kwargs = {
            "skip_special_tokens": False,
            "tokenizer": self.tokenizer,
        }
        repr = f"Question #{row.get('question.idx', None)}\n"

        repr += get_separator("-") + "\n"
        repr += "* Question-answer:" + "\n"
        idx = row["answer.target"]
        for i, an in enumerate(row["question.input_ids"]):
            an_style = "green" if idx == i else "white"
            line = (
                f"   - ({'x' if idx == i else ' '}) "
                f"{pretty_decode(an, **decode_kwargs, only_text=False, style=an_style)}\n"
            )
            repr += line

        return repr
