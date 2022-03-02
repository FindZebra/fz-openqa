import logging
from functools import partial
from typing import Any
from typing import Dict
from typing import Optional

import dill  # type: ignore
import rich
from datasets import DatasetDict
from datasets import load_dataset
from datasets.arrow_dataset import concatenate_datasets

from ..pipelines.collate.field import CollateField
from ..pipes.answer_options import ConcatTextFields
from ..pipes.control.condition import In
from ..pipes.min_length import MinLength
from ..pipes.nesting import ApplyAsFlatten
from ..pipes.nesting import Expand
from ..pipes.nesting import Nested
from ..pipes.tokenizer import QueryExpansionPipe
from ..utils.dataset import format_size_difference
from .hf_dataset import HfDatasetBuilder
from fz_openqa.datamodules.generators import medqa
from fz_openqa.datamodules.generators import quality
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

QA_DATASETS = {
    "medqa-us": (medqa.__file__, "us"),
    "medqa-tw": (medqa.__file__, "tw"),
    "quality": (quality.__file__, "questions"),
}


class QaBuilder(HfDatasetBuilder):
    # HuggingFace dataset id or local path to script
    # these values are set dynamically in the __init__
    dset_script_path_or_id = None
    dset_name = None

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
        self,
        *args,
        min_answer_length: Optional[int] = None,
        n_query_tokens: int = 1,
        n_answer_tokens: int = 1,
        query_expansion: Optional[int] = None,
        dset_name: str = "medqa-us",
        **kwargs,
    ):
        super(QaBuilder, self).__init__(*args, **kwargs)
        self.min_answer_length = min_answer_length
        self.query_expansion = query_expansion
        self.n_query_tokens = n_query_tokens
        self.n_answer_tokens = n_answer_tokens

        # set the dataset attributes
        self.dset_script_path_or_id = QA_DATASETS[dset_name][0]
        self.dset_name = QA_DATASETS[dset_name][1]

    def load_base_dataset(self) -> DatasetDict:
        """
        Loads the base dataset. Multiple dataset names can be passed
        using "+" as a separator. e.g. "tw+us"
        """
        dset_names = sorted(self.dset_name.split("+"))

        rich.print(f">> loading: {self.dset_script_path_or_id} with {dset_names}")

        kwargs = {"cache_dir": self.cache_dir}
        dsets = [load_dataset(self.dset_script_path_or_id, name=n, **kwargs) for n in dset_names]

        if len(dsets) == 1:
            return dsets[0]
        else:
            dsets_dict = DatasetDict()
            for split in dsets[0].keys():
                split_dsets = concatenate_datasets([d[split] for d in dsets])
                dsets_dict[split] = split_dsets

            return dsets_dict

    def filter_dataset(self, dataset: HfDataset) -> HfDataset:
        """Apply filter operation to the dataset and return"""

        # filter out answers that are too short
        if self.min_answer_length is not None:
            logger.info(f"Filtering out answers shorter than {self.min_answer_length} characters")
            lengths = {k: len(d) for k, d in dataset.items()}
            dataset = dataset.filter(MinLength("answer.text", self.min_answer_length))
            logger.info(format_size_difference(lengths, dataset))

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
            spec_tokens=self.n_answer_tokens * [ANS_TOKEN],
            shape=[-1, self.n_options],
        )

    def get_question_tokenizer_pipe(self):
        """create a Pipe to tokenize the questions."""

        if self.query_expansion is not None:
            query_expansion_pipe = QueryExpansionPipe(
                question_length=self.query_expansion, tokenizer=self.tokenizer, update=True
            )
        else:
            query_expansion_pipe = None

        return Sequential(
            FormatAndTokenize(
                prefix="question.",
                text_formatter=self.text_formatter,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                add_encoding_tokens=self.add_encoding_tokens,
                add_special_tokens=self.add_special_tokens,
                spec_tokens=self.n_query_tokens * [QUERY_TOKEN],
                shape=None,
            ),
            query_expansion_pipe,
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


class ConcatQaBuilder(QaBuilder):
    """A MedQa dataset with concatenated questions and answers"""

    # nesting level of the question field
    nesting_level = 1

    # name of the attributes that will be converted to
    # tensors in the preprocessing function
    pt_attributes = [
        "question.input_ids",
        "question.attention_mask",
        "question.document_idx",
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
        # register features that also need to be expanded to match the concatenated shape
        additional_question_features = ["question.document_idx"]

        # register the tokens that prefix the question
        q_start_tokens = []
        if self.add_special_tokens:
            q_start_tokens.append(self.tokenizer.sep_token)
        if self.add_encoding_tokens:
            q_start_tokens.extend(self.n_query_tokens * [QUERY_TOKEN])

        if len(q_start_tokens) > 0:
            add_spec_tokens_pipe = Apply(
                {"question.text": partial(add_spec_token, q_start_tokens)},
                element_wise=True,
            )
        else:
            add_spec_tokens_pipe = None

        # return the final pipe
        return Sequential(
            self.text_formatter.copy(text_key=["question.text", "answer.text"], update=True),
            add_spec_tokens_pipe,
            Expand(
                axis=1,
                n=self.n_options,
                update=True,
                input_filter=In(["question.text", *additional_question_features]),
            ),
            ApplyAsFlatten(
                ConcatTextFields(keys=["answer.text", "question.text"], new_key="question.text"),
                level=1,
                input_filter=In(["question.text", "answer.text"]),
                update=True,
            ),
            input_filter=In(["question.text", "answer.text", *additional_question_features]),
        )

    def get_qa_tokenizer_pipe(self):

        if self.query_expansion is not None:
            query_expansion_pipe = Nested(
                QueryExpansionPipe(
                    question_length=self.query_expansion, tokenizer=self.tokenizer, update=True
                ),
                level=1,
            )
        else:
            query_expansion_pipe = None

        return Sequential(
            FormatAndTokenize(
                prefix="question.",
                text_formatter=None,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                add_encoding_tokens=self.add_encoding_tokens,
                add_special_tokens=self.add_special_tokens,
                spec_tokens=self.n_answer_tokens * [ANS_TOKEN],
                shape=[-1, self.n_options],
            ),
            query_expansion_pipe,
        )

    def _get_collate_pipe(self):
        # get the raw text questions, extract and collate
        return Parallel(
            CollateField(
                "question",
                exclude=["input_ids", "attention_mask"],
                to_tensor=["document_idx"],
                level=0,
                id="collate-questions",
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
