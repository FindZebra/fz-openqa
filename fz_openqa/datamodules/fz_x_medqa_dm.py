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

    def tokenize_examples(
        self,
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
        output = {
            "question.text": examples["question"],
            "document.text": examples["document"],
        }
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
        self.pt_attributes += ["answer_idx", "rank", "is_positive"]
        dataset.set_format(
            type="torch", columns=self.pt_attributes, output_all_columns=False
        )
        return dataset

    def filter_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply filtering operations"""
        if self.filter_gold:
            dataset = dataset.filter(
                lambda x: x["rank"] == 0 and x["is_positive"]
            )

        return dataset

    def collate_fn(
        self, batch: Any
    ) -> Union[BatchEncoding, Dict[str, torch.Tensor]]:
        """The function that is used to merge examples into a batch.
        Concatenating sequences with different length requires padding them.
        Returns a dictionary with attributes:
        output = {
                rank: tensor of shape [N,],
                question.input_ids: tensor of shape [N, T],
                question.attention_mask: tensor of shape [N, T],
                document.input_ids: tensor of shape [N, T],
                document.attention_mask: tensor of shape [N, T],
                answer_choices.input_ids: tensor of shape [N, N_a, T]
                answer_choices.attention_mask: tensor of shape [N, N_a, T]
                answer_idx: tensor of shape [N,]
        }
        """
        attrs = ["input_ids", "attention_mask"]
        output = {}

        # answer_idx & rank attributes
        output["answer_idx"] = torch.tensor([b["answer_idx"] for b in batch])
        output["rank"] = torch.tensor([b["rank"] for b in batch])
        output["is_positive"] = torch.tensor([b["is_positive"] for b in batch])

        # documents and questions
        for key in ["document", "question"]:
            doc_encoding = self.tokenizer.pad(
                [
                    {
                        k.replace(f"{key}.", ""): v
                        for k, v in b.items()
                        if f"{key}." in k
                    }
                    for b in batch
                ]
            )
            for attr, tsr in doc_encoding.items():
                output[f"{key}.{attr}"] = tsr

        # merge answers:
        ans_cols = ["answer_0", "answer_1", "answer_2", "answer_3"]
        ans_encoding = self.tokenizer.pad(
            {
                attr: [b[f"{ans}.{attr}"] for ans in ans_cols for b in batch]
                for attr in attrs
            }
        )
        for k, v in ans_encoding.items():
            n_pts, n_ans = len(ans_cols), len(output["answer_idx"])
            output[f"answer_choices.{k}"] = (
                v.view(n_pts, n_ans, -1).permute(1, 0, 2).contiguous()
            )

        return output

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
        rich.print(
            f"* Document (rank={example['rank']}, is_positive={example['is_positive']})"
        )
        rich.print(
            self.repr_ex(example, "document.input_ids", **decode_kwargs)
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
