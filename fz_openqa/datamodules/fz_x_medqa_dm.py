import random
import shutil
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import datasets
import rich
import torch
from datasets import Split
from transformers import BatchEncoding
from transformers import PreTrainedTokenizerFast

from .base_dm import BaseDataModule
from .base_dm import HgDataset
from .collate import collate_and_pad_attributes
from .collate import collate_answer_options
from .collate import collate_nested_examples
from .collate import collate_simple_attributes_by_key
from .collate import extract_and_collate_attributes_as_list
from .datasets import fz_x_medqa
from .utils import add_spec_token
from .utils import nested_list
from fz_openqa.tokenizers.static import ANS_TOKEN
from fz_openqa.tokenizers.static import DOC_TOKEN
from fz_openqa.tokenizers.static import QUERY_TOKEN
from fz_openqa.utils.datastruct import Batch

PT_SIMPLE_ATTRIBUTES = [
    "answer.target",
    "answer.n_options",
    "document.rank",
    "document.is_positive",
    "question.idx",
    "idx",
]


class FZxMedQADataModule(BaseDataModule):
    """A PyTorch Lightning DataModule wrapping the FZxMedQA dataset."""

    dset_script_path_or_id = (
        fz_x_medqa.__file__
    )  # HuggingFace dataset id or local path to script
    split_ids = [
        datasets.Split.TRAIN,
        datasets.Split.VALIDATION,
        datasets.Split.TEST,
    ]  # split names
    text_fields = ["question", "answer", "document"]
    pt_attributes = None  # to be generated

    def __init__(self, *, filter_gold: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.filter_gold = filter_gold

    @staticmethod
    def tokenize_examples(
        examples: Dict[str, List[Any]],
        *,
        tokenizer: PreTrainedTokenizerFast,
        max_length: Optional[int],
        add_encoding_tokens: bool = True,
    ) -> Union[Dict, BatchEncoding]:
        """Tokenize a batch of examples and truncate if `max_length` is provided.
        examples = {
            attribute_name: list of attribute values
        }
        the output if of the form:
        output = {
            document.input_ids: [e.document.tokens for e in examples]
            document.attention_mask: [e.document.mask for e in examples]
            question.input_ids: [e.question.tokens for e in examples]
            question.attention_mask: [e.question.mask for e in examples]
            answer_0.input_ids: [e.answer_0.tokens for e in examples]
            answer_0.attention_mask: [e.answer_0.mask for e in examples]
            answer_1.input_ids: [e.answer_1.tokens for e in examples]
            answer_1.attention_mask: [e.answer_1.mask for e in examples]
            [...]
        }

        """
        tokenizer_kwargs = {
            "max_length": max_length,
            "return_token_type_ids": False,
            "add_special_tokens": True,
            "truncation": max_length is not None,
        }

        # process questions and documents
        questions = (
            list(
                map(partial(add_spec_token, QUERY_TOKEN), examples["question"])
            )
            if add_encoding_tokens
            else examples["question"]
        )
        documents = (
            list(map(partial(add_spec_token, DOC_TOKEN), examples["document"]))
            if add_encoding_tokens
            else examples["document"]
        )
        q_encodings = tokenizer(questions, **tokenizer_kwargs)
        d_encodings = tokenizer(documents, **tokenizer_kwargs)

        # prepare the output
        output = {}

        # append the "question" and "document" prefix to the
        # "input_ids" and "attention masts from the q/d_encodings
        # and store them in output.
        for data, prefix in zip(
            [q_encodings, d_encodings], ["question", "document"]
        ):
            for k, v in data.items():
                output[f"{prefix}.{k}"] = v

        # process answers
        add_answ_token = (
            partial(add_spec_token, ANS_TOKEN)
            if add_encoding_tokens
            else lambda x: x
        )
        output["answer.n_options"] = [len(ex) for ex in examples["answer"]]
        answer_encodings = tokenizer(
            [
                add_answ_token(choice)
                for choices in examples["answer"]
                for choice in choices
            ],
            **tokenizer_kwargs,
        )
        assert all(
            x == output["answer.n_options"][0]
            for x in output["answer.n_options"]
        )
        output.update(
            **{
                f"answer.{k}": nested_list(
                    v, stride=output["answer.n_options"][0]
                )
                for k, v in answer_encodings.items()
            }
        )

        return output

    def preprocess_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply processing steps to the dataset. Tokenization and formatting as PyTorch tensors"""
        # tokenize and format as PyTorch tensors
        fn = partial(
            self.tokenize_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            add_encoding_tokens=self.add_encoding_tokens,
        )
        dataset = dataset.map(
            fn, batched=True, num_proc=self.num_proc, desc="Tokenizing"
        )

        # rename text attributes
        for key in ["document", "question", "answer"]:
            dataset = dataset.rename_column(key, f"{key}.text")

        # transform attributes to tensors
        attrs = ["input_ids", "attention_mask"]
        columns = ["question", "document", "answer"]
        self.pt_attributes = [
            c
            for c in dataset.column_names["train"]
            if (any(a in c for a in attrs) and any(a in c for a in columns))
        ]
        self.pt_attributes += PT_SIMPLE_ATTRIBUTES
        dataset.set_format(
            type="torch", columns=self.pt_attributes, output_all_columns=True
        )

        return dataset

    def filter_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply filtering operations"""
        if self.filter_gold:
            dataset = dataset.filter(
                lambda x: x["document.rank"] == 0 and x["document.is_positive"]
            )

        return dataset

    @staticmethod
    def filter_question_id(ids: List[int], row: Dict[str, Any]) -> bool:
        return row["question.idx"] in ids

    @staticmethod
    def take_subset(dataset: HgDataset) -> HgDataset:
        """Take a subset of the dataset and return."""
        subset_size = {Split.TRAIN: 5, Split.VALIDATION: 2, Split.TEST: 2}
        for key, dset in dataset.items():
            questions_ids = dset["question.idx"]
            selected_ids = random.sample(questions_ids, k=subset_size[key])
            fn = partial(FZxMedQADataModule.filter_question_id, selected_ids)
            dataset[key] = dset.filter(fn)

        return dataset

    def collate_fn(self, examples: Any) -> Batch:
        return self.collate_fn_(self.tokenizer, examples)

    @staticmethod
    def collate_fn_(
        tokenizer: PreTrainedTokenizerFast,
        examples: Any,
    ) -> Batch:
        """The function that is used to merge examples into a batch.
        Concatenating sequences with different length requires padding them.

        The input data must be of the following structure:
        ```yaml
            - example_1:
                - sub_example_1:
                    - question.idx: a
                    - document.idx: x
                - sub_example_2:
                    - question.idx: a
                    - document.idx: y
            - example_2:
                - sub_example_1:
                    - question.idx: b
                    - document.idx: y
                - sub_example_2:
                    - question.idx: a
                    - document.idx: z
        ```
        If the input data is a List of examples, it will be converted bellow as a list of list of examples.

        Returns a dictionary with attributes:
        output = {
                question.idx: tensor of shape [N,],
                question.text: list of N texts
                question.input_ids: tensor of shape [N, T],
                question.attention_mask: tensor of shape [N, T],
                document.text: nested list of [N, N_docs] texts
                document.input_ids: tensor of shape [N, N_doc,  T],
                document.attention_mask: tensor of shape [N, N_doc, T],
                document.rank: tensor of shape [N, N_doc],
                document.is_positive: tensor of shape [N, N_doc],
                answer.text: nested list of [N, N_a] texts
                answer.input_ids: tensor of shape [N, N_a, T]
                answer.attention_mask: tensor of shape [N, N_a, T]
                answer.target: tensor of shape [N,]
                answer.n_options: tensor of hsape [N,]
        }
        """
        output = {}

        # convert as List of Lists if that's not the case (general case)
        if not isinstance(examples[0], list):
            examples = [[ex] for ex in examples]

        # collate the question and answers using the first example of each batch element
        first_examples = [ex[0] for ex in examples]
        output.update(
            **FZxMedQADataModule.collate_qa(
                first_examples, tokenizer=tokenizer
            )
        )

        # collate documents
        output.update(
            **collate_nested_examples(
                examples, tokenizer=tokenizer, key="document"
            )
        )

        return output

    @staticmethod
    def collate_qa(
        examples: List[Batch], *, tokenizer: PreTrainedTokenizerFast
    ) -> Batch:
        """collate the question and answer data"""

        # get the raw text questions, extract and collate
        examples, output = extract_and_collate_attributes_as_list(
            examples, attribute="text", key="question"
        )

        # collate simple attributes
        for k in ["idx", "answer.target", "answer.n_options"]:
            output[k] = collate_simple_attributes_by_key(
                examples, key=k, extract=True
            )

        # collate the questions attributes (question.input_ids, question.idx, ...)
        output.update(
            **collate_and_pad_attributes(
                examples, tokenizer=tokenizer, key="question", exclude=".text"
            )
        )

        # collate answer options
        output.update(**collate_answer_options(examples, tokenizer=tokenizer))

        return output

    def display_one_sample(self, example: Dict[str, torch.Tensor]):
        """Decode and print one example from the batch"""
        decode_kwargs = {"skip_special_tokens": False}
        console_width, _ = shutil.get_terminal_size()
        print("=== Sample ===")
        print(console_width * "-")
        print("* Question:")
        rich.print(
            self.repr_ex(example, "question.input_ids", **decode_kwargs)
        )

        print(console_width * "-")
        rich.print("* Documents: ")
        for k in range(example["document.input_ids"].shape[0]):
            rich.print(
                f"-- document [magenta]{k}[/magenta] (is_positive={example['document.is_positive'][k]}, rank={example['document.rank'][k]}) --"
            )
            rich.print(
                self.repr_ex(
                    {"input_ids": example["document.input_ids"][k]},
                    "input_ids",
                    **decode_kwargs,
                )
            )
        print(console_width * "-")
        print("* Answer Choices:")
        idx = example["answer.target"]
        for i, an in enumerate(example["answer.input_ids"]):
            print(
                f"   - [{'x' if idx == i else ' '}] "
                f"{self.tokenizer.decode(an, **decode_kwargs).replace('[PAD]', '').strip()}"
            )
        print(console_width * "=")
