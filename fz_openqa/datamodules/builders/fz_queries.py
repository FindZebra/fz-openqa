import logging
from functools import partial
from typing import Any
from typing import Dict

from fz_openqa.datamodules.builders.qa import QaBuilder
from fz_openqa.datamodules.generators import fz_queries
from fz_openqa.datamodules.pipelines.collate.field import CollateField
from fz_openqa.datamodules.pipes.min_length import MinLength
from fz_openqa.datamodules.utils.dataset import format_size_difference
from fz_openqa.datamodules.utils.transformations import set_row_idx
from fz_openqa.datamodules.utils.typing import HfDataset
from fz_openqa.utils.pretty import pretty_decode

logger = logging.getLogger(__name__)


class FzQueriesBuilder(QaBuilder):
    dset_script_path_or_id = fz_queries.__file__

    # nesting level of the question field
    nesting_level = 0

    # name of the attributes that will be converted to
    # tensors in the preprocessing function
    pt_attributes = [
        "question.input_ids",
        "question.attention_mask",
        "question.idx",
    ]

    # number of data points per subset train/val/test
    subset_size = [1000, 100, 100]

    # number of options
    n_options = None

    # output columns
    column_names = [
        "question.cui",
        "question.text",
        "question.input_ids",
        "question.attention_mask",
    ]

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
                self.get_question_tokenizer_pipe(),
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

    def _get_collate_pipe(self):
        # get the raw text questions, extract and collate
        return CollateField("question", tokenizer=self.tokenizer, level=0, id="collate-questions")

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

        return u
