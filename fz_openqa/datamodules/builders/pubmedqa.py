from functools import partial
from typing import Any
from typing import Dict
from typing import Optional

import datasets
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

from ..pipelines.collate.field import CollateField
from .hf_dataset import HfDatasetBuilder
from fz_openqa.datamodules.pipelines.preprocessing.text import FormatAndTokenize
from fz_openqa.datamodules.pipes.meta import Parallel
from fz_openqa.datamodules.utils.transformations import set_row_idx
from fz_openqa.datamodules.utils.typing import HfDataset
from fz_openqa.tokenizers.static import ANS_TOKEN
from fz_openqa.tokenizers.static import QUERY_TOKEN
from fz_openqa.utils.pretty import pretty_decode


class PubMedQaBuilder(HfDatasetBuilder):
    # HuggingFace dataset id
    dset_script_path_or_id = "pqa_labeled"

    # text field from the raw datasets that should be tokenized
    text_field = "question"

    # attributes to be converted to tensors in preprocessing
    pt_attributes = [
        "question.input_ids",
        "question.attention_mask",
        "question.idx",
        "question.row_idx",
        "answer.input_ids",
        "answer.attention_mask",
        "answer.target",
    ]

    # number of data points per subset train/val/test
    subset_size = [800, 100, 100]

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

    def __init__(self, *args, n_query_tokens: int = 1, **kwargs):
        super(PubMedQaBuilder, self).__init__(*args, **kwargs)
        self.n_query_tokens = n_query_tokens

    def load_base_dataset(self) -> DatasetDict:
        """Load the base HuggingFace dataset and split into train, validation, test """
        dataset = load_dataset(
            "pubmed_qa", self.dset_script_path_or_id, cache_dir=self.cache_dir
        )  # todo: make more dynamic

        # todo: make renaming of columns more efficient
        new_column_names = {
            "question": "question.text",
            "pubid": "question.idx",
            "long_answer": "answer.text",
            "final_decision": "answer.target",
            "context": "document.text",
        }
        for k in new_column_names.keys():
            dataset = dataset.rename_column(k, new_column_names[k])

        dataset = datasets.Dataset.train_test_split(dataset["train"], train_size=0.8)
        dataset_eval = datasets.Dataset.train_test_split(dataset["test"], test_size=0.5)
        dataset["validation"] = dataset_eval["train"]
        dataset["test"] = dataset_eval["test"]

        return dataset

    def filter_dataset(self, dataset: HfDataset) -> HfDataset:
        """Apply filter operation to the dataset and return"""
        return dataset

    def preprocess_dataset(self, dataset: HfDataset) -> HfDataset:
        """Apply processing steps to the dataset.
        Tokenization and formatting as PyTorch tensors"""

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
                row["input_ids"],
                **decode_kwargs,
                style="deep_sky_blue3",
            )
            + "\n"
        )
        return u


class PubMedQaArtificialBuilder(PubMedQaBuilder):
    subset_size = [20]
    dset_script_path_or_id = "pqa_artificial"
