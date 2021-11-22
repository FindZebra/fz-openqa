import logging
from functools import partial
from typing import Any
from typing import Dict

import dill  # type: ignore
from datasets import DatasetDict
from datasets import load_dataset

from ..pipelines.collate.field import CollateField
from .hf_dataset import HfDatasetBuilder
from fz_openqa.datamodules.generators import medqa
from fz_openqa.datamodules.pipelines.collate import CollateAsTensor
from fz_openqa.datamodules.pipelines.collate import CollateTokens
from fz_openqa.datamodules.pipelines.preprocessing import FormatAndTokenize
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes import Parallel
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.utils.transformations import set_row_idx
from fz_openqa.datamodules.utils.typing import HfDataset
from fz_openqa.tokenizers.static import ANS_TOKEN
from fz_openqa.tokenizers.static import QUERY_TOKEN
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pretty_decode

logger = logging.getLogger(__name__)


class MedQaBuilder(HfDatasetBuilder):
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

    def load_base_dataset(self) -> DatasetDict:
        """Load the base HuggingFace dataset."""
        return load_dataset(self.dset_script_path_or_id, cache_dir=self.cache_dir)

    def filter_dataset(self, dataset: HfDataset) -> HfDataset:
        """Apply filter operation to the dataset and return"""
        return dataset

    def preprocess_dataset(self, dataset: HfDataset) -> HfDataset:
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
            spec_tokens=QUERY_TOKEN,
            shape=None,
        )

    def get_collate_pipe(self):
        # get the raw text questions, extract and collate
        return Parallel(
            CollateField("question", tokenizer=self.tokenizer, level=0),
            CollateField("answer", level=0, exclude=["input_ids", "attention_mask"]),
            CollateField(
                "answer",
                tokenizer=self.tokenizer,
                level=1,
                exclude=["synonyms", "target", "cui", "synonyms", "text"],
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
