from typing import Optional

import datasets
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from datasets.splits import Split

from .hf_dataset import HfDatasetBuilder
from fz_openqa.datamodules.pipelines.preprocessing.text import FormatAndTokenize
from fz_openqa.datamodules.pipes.meta import Parallel
from fz_openqa.datamodules.utils.typing import HfDataset
from fz_openqa.tokenizers.static import ANS_TOKEN
from fz_openqa.tokenizers.static import QUERY_TOKEN


class PubMedQaBuilder(HfDatasetBuilder):
    # HuggingFace dataset id
    dset_script_path_or_id = "pqa_labeled"

    # attributes to be converted to tensors in preprocessing
    pt_attributes = [
        "answer.label",
        "answer.input_ids",
        "answer.attention_mask",
        "question.input_ids",
        "question.attention_mask",
        "pubid",
    ]

    # number of data points per subset train/val/test
    subset_size = [1000, 100, 100]

    # output columns
    column_names = [
        "answer.label",
        "answer.input_ids",
        "answer.attention_mask",
        "question.text",
        "question.input_ids",
        "question.attention_mask",
    ]

    def __init__(self, *args, n_query_tokens: int = 1, **kwargs):
        super(PubMedQaBuilder, self).__init__(*args, **kwargs)
        self.n_query_tokens = n_query_tokens

    def load_base_dataset(self) -> DatasetDict:
        dataset = load_dataset("pubmed_qa", self.dset_script_path_or_id)  # todo: make more dynamic
        dataset = datasets.Dataset.train_test_split(
            dataset["train"], train_ratio=0.8
        )  # todo: use input parameter split
        dataset_eval = datasets.Dataset.train_test_split(dataset["test"], train_ratio=0.5)
        dataset["validation"] = dataset_eval["train"]
        dataset["test"] = dataset_eval["test"]

        return dataset

    def preprocess_dataset(self, dataset: HfDataset) -> HfDataset:

        # Tokenize question and answer
        if self.tokenizer:
            dataset = dataset.map(
                Parallel(self.get_question_tokenizer_pipe(), self.get_answer_tokenizer_pipe()),
                batched=True,
                num_proc=self.num_proc,
                desc="Tokenization of questions and answers",
            )

        return dataset

    def get_answer_tokenizer_pipe(self):
        return FormatAndTokenize(
            prefix="answer.",
            text_formatter=None,
            tokenizer=self.tokenizer,
            max_length=None,
            add_encoding_tokens=self.add_encoding_tokens,
            spec_tokens=ANS_TOKEN,
            shape=None,
        )

    def get_question_tokenizer_pipe(self):
        """create a Pipe to tokenize the questions."""
        return FormatAndTokenize(
            prefix="question.",
            text_formatter=None,
            tokenizer=self.tokenizer,
            max_length=None,
            add_encoding_tokens=self.add_encoding_tokens,
            add_special_tokens=self.add_special_tokens,
            spec_tokens=self.n_query_tokens * [QUERY_TOKEN],
            shape=None,
        )
