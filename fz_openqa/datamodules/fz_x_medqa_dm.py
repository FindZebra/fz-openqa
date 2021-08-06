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
from transformers import BatchEncoding
from transformers import PreTrainedTokenizerFast

from .base_dm import BaseDataModule
from .base_dm import HgDataset
from .datasets import fz_x_medqa
from fz_openqa.tokenizers.static import ANS_TOKEN
from fz_openqa.tokenizers.static import DOC_TOKEN
from fz_openqa.tokenizers.static import QUERY_TOKEN
from fz_openqa.utils.datastruct import Batch

PT_SIMPLE_ATTRIBUTES = [
    "answer_idx",
    "document.rank",
    "document.is_positive",
    "question.idx",
    "idx",
]
ENCODING_ATTRIBUTES = ["input_ids", "attention_mask"]
DEFAULT_ANSWER_COLUMNS = ["answer_0", "answer_1", "answer_2", "answer_3"]


def add_spec_token(
    special_token: str,
    text: str,
):
    """
    This functions append a special token to a text such that output = special_token+text.
    The pretrained tokenizer with registered special tokens will encode the output as:
    [CLS][SPEC][ text tokens ][SEP]
    """
    assert special_token in [QUERY_TOKEN, ANS_TOKEN, DOC_TOKEN]
    return f"{special_token}{text}"


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
        questions = list(
            map(partial(add_spec_token, QUERY_TOKEN), examples["question"])
        )
        documents = list(
            map(partial(add_spec_token, DOC_TOKEN), examples["document"])
        )
        q_encodings = tokenizer(questions, **tokenizer_kwargs)
        d_encodings = tokenizer(documents, **tokenizer_kwargs)

        # store the raw text
        output = {
            "question.text": examples["question"],
            "document.text": examples["document"],
        }

        # append the "question" and "document" prefix to the
        # "input_ids" and "attention masts from the q/d_encodings
        # and store them in output.
        for data, prefix in zip(
            [q_encodings, d_encodings], ["question", "document"]
        ):
            for k, v in data.items():
                output[f"{prefix}.{k}"] = v

        # process answers
        add_answ_token = partial(add_spec_token, ANS_TOKEN)
        n_choices = len(examples["answer_choices"][0])
        answer_encodings = [
            tokenizer(
                [add_answ_token(ans[n]) for ans in examples["answer_choices"]],
                **tokenizer_kwargs,
            )
            for n in range(n_choices)
        ]

        for idx, data in enumerate(answer_encodings):
            for k, v in data.items():
                output[f"answer_{idx}.{k}"] = v

        return output

    def preprocess_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply processing steps to the dataset. Tokenization and formatting as PyTorch tensors"""
        # tokenize and format as PyTorch tensors
        fn = partial(
            self.tokenize_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        dataset = dataset.map(
            fn, batched=True, num_proc=self.num_proc, desc="Tokenizing"
        )
        # transform attributes to tensors
        attrs = ["input_ids", "attention_mask"]
        columns = ["question", "document", "answer_"]
        self.pt_attributes = [
            c
            for c in dataset.column_names["train"]
            if (any(a in c for a in attrs) and any(a in c for a in columns))
        ]
        self.pt_attributes += PT_SIMPLE_ATTRIBUTES
        dataset.set_format(
            type="torch", columns=self.pt_attributes, output_all_columns=False
        )
        return dataset

    def filter_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply filtering operations"""
        if self.filter_gold:
            dataset = dataset.filter(
                lambda x: x["document.rank"] == 0 and x["document.is_positive"]
            )

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
                question.input_ids: tensor of shape [N, T],
                question.attention_mask: tensor of shape [N, T],
                document.input_ids: tensor of shape [N, N_doc,  T],
                document.attention_mask: tensor of shape [N, N_doc, T],
                document.rank: tensor of shape [N, N_doc],
                document.is_positive: tensor of shape [N, N_doc],
                answer_choices.input_ids: tensor of shape [N, N_a, T]
                answer_choices.attention_mask: tensor of shape [N, N_a, T]
                answer_idx: tensor of shape [N,]
        }
        """

        # convert as List of Lists if that's not the case (general case)
        if not isinstance(examples[0], list):
            examples = [[ex] for ex in examples]

        # collate the question and answers using the first example of each batch element
        first_examples = [ex[0] for ex in examples]
        output = FZxMedQADataModule.collate_qa(
            first_examples, tokenizer=tokenizer
        )

        # collate documents
        output.update(
            **FZxMedQADataModule.collate_documents(
                examples, tokenizer=tokenizer
            )
        )
        return output

    @staticmethod
    def collate_qa(
        examples: List[Batch], *, tokenizer: PreTrainedTokenizerFast
    ) -> Batch:
        """collate the question and answer data"""
        # collate the questions (text)
        output = FZxMedQADataModule.collate_text(
            examples, tokenizer=tokenizer, prefix="question."
        )

        # collate answer options
        output.update(
            **FZxMedQADataModule.collate_answer_options(
                examples, tokenizer=tokenizer
            )
        )

        # collate simple attributes
        for k in ["question.idx", "idx", "answer_idx"]:
            output[k] = FZxMedQADataModule.collate_simple_attribute(
                examples, key=k
            )

        return output

    @staticmethod
    def collate_documents(examples, *, tokenizer: PreTrainedTokenizerFast):

        # flatten examples
        flattened_examples = [sub_ex for ex in examples for sub_ex in ex]

        # encode document text
        document_output = FZxMedQADataModule.collate_text(
            flattened_examples, tokenizer=tokenizer, prefix="document."
        )

        # encode simple attributes
        for k in ["document.rank", "document.is_positive"]:
            document_output[k] = FZxMedQADataModule.collate_simple_attribute(
                flattened_examples, key=k
            )

        # reshape document data as shape [batch_size, n_docs, ...]
        n_docs = len(examples[0])
        batch_size = len(examples)
        return {
            k: v.view(batch_size, n_docs, *v.shape[1:])
            for k, v in document_output.items()
        }

    @staticmethod
    def collate_simple_attribute(examples, *, key: str):
        return torch.tensor([ex[key] for ex in examples])

    @staticmethod
    def collate_answer_options(
        examples: List[Batch],
        *,
        tokenizer: PreTrainedTokenizerFast,
        answer_columns: Optional[List] = None,
        input_attributes: Optional[List[str]] = None,
    ) -> Batch:
        ans_cols = answer_columns or DEFAULT_ANSWER_COLUMNS
        input_attributes = input_attributes or ENCODING_ATTRIBUTES
        batch_size = len(examples)
        n_ans = len(ans_cols)
        ans_encoding = tokenizer.pad(
            {
                attr: [
                    ex[f"{ans}.{attr}"] for ans in ans_cols for ex in examples
                ]
                for attr in input_attributes
            }
        )
        output = {}
        for k, v in ans_encoding.items():
            output[f"answer_choices.{k}"] = (
                v.view(n_ans, batch_size, -1).permute(1, 0, 2).contiguous()
            )

        return output

    @staticmethod
    def collate_text(
        examples: List[Batch],
        *,
        tokenizer: PreTrainedTokenizerFast,
        prefix: str = "",
        input_attributes: Optional[List[str]] = None,
    ) -> Batch:
        """
        Collate a List of examples of text data into a single batch.
        """

        # get the list of attributes to select from each batch
        input_attributes = input_attributes or ENCODING_ATTRIBUTES
        input_attributes = [f"{prefix}{k}" for k in input_attributes]

        # encode the batches
        doc_encoding = tokenizer.pad(
            [
                {
                    k.replace(f"{prefix}", ""): v
                    for k, v in b.items()
                    if k in input_attributes
                }
                for b in examples
            ]
        )

        # append the prefix back and return
        return {f"{prefix}{attr}": tsr for attr, tsr in doc_encoding.items()}

    def display_one_sample(self, example: Dict[str, torch.Tensor]):
        """Decode and print one example from the batch"""
        decode_kwargs = {"skip_special_tokens": True}
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
        idx = example["answer_idx"]
        for i, an in enumerate(example["answer_choices.input_ids"]):
            print(
                f"   - [{'x' if idx == i else ' '}] "
                f"{self.tokenizer.decode(an, **decode_kwargs)}"
            )
        print(console_width * "=")
