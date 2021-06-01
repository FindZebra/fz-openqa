import shutil
from functools import partial
from typing import *

import datasets
import torch
from transformers import BatchEncoding, PreTrainedTokenizerFast

from .base_dm import BaseDataModule, HgDataset
from .datasets import fz_x_medqa

class FZxMedQA(BaseDataModule):
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

    def encode(
            self,
            examples: Dict[str, Any],
            *,
            tokenizer: PreTrainedTokenizerFast,
            max_length: Optional[int],
    ) -> Union[Dict, BatchEncoding]:

        tokenizer_kwargs = {
            "max_length": max_length,
            "return_token_type_ids": False,
            "add_special_tokens": True,
            "truncation": max_length is not None,
        }
        # process questions and documents
        q_encodings = tokenizer(examples["question"], **tokenizer_kwargs)
        d_encodings = tokenizer(examples["document"], **tokenizer_kwargs)
        output = {
            "question.text": examples["question"],
            "document.text": examples["document"],
        }
        for data, prefix in zip([q_encodings, d_encodings], ["question", "document"]):
            for k, v in data.items():
                output[f"{prefix}.{k}"] = v

        # process answers
        n_choices = len(examples["answer_choices"][0])
        answer_encodings = [
            tokenizer(
                [ans[n] for ans in examples["answer_choices"]], **tokenizer_kwargs
            )
            for n in range(n_choices)
        ]

        for idx, data in enumerate(answer_encodings):
            for k, v in data.items():
                output[f"answer_{idx}.{k}"] = v

        return output

    def preprocess_dataset(self, dataset: HgDataset) -> HgDataset:
        # tokenize and format as PyTorch tensors
        fn = partial(self.encode, tokenizer=self.tokenizer, max_length=self.max_length)
        dataset = dataset.map(fn, batched=True)
        # transform attributes to tensors
        attrs = ["input_ids", "attention_mask"]
        columns = ["question", "document", "answer_"]
        self.pt_attributes = [
            c
            for c in dataset.column_names["train"]
            if (any(a in c for a in attrs) and any(a in c for a in columns))
        ]
        self.pt_attributes += ["answer_idx", "is_gold"]
        dataset.set_format(
            type="torch", columns=self.pt_attributes, output_all_columns=False
        )
        return dataset

    def filter_dataset(self, dataset: HgDataset) -> HgDataset:
        # keep only gold passages
        return dataset.filter(lambda x: x["is_gold"])

    def collate_fn(self, batch: Any) -> Union[BatchEncoding, Dict[str, torch.Tensor]]:
        attrs = ["input_ids", "attention_mask"]
        output = {}

        # answer_idx & is_gold attributes
        output["answer_idx"] = torch.tensor([b["answer_idx"] for b in batch])
        output["is_gold"] = torch.tensor([b["is_gold"] for b in batch])

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
        decode_kwargs = {'skip_special_tokens': True}
        console_width, _ = shutil.get_terminal_size()
        print("=== Sample ===")
        print(console_width * "-")
        print("* Question:")
        print(console_width * "-")
        print(self.tokenizer.decode(example['question.input_ids'], **decode_kwargs))
        print(f"* Document (is_gold={example['is_gold']})")
        print(console_width * "-")
        print(self.tokenizer.decode(example['document.input_ids'], **decode_kwargs))
        print("* Answer Choices:")
        print(console_width * "-")
        idx = example['answer_idx']
        for i, an in enumerate(example['answer_choices.input_ids'], **decode_kwargs):
            print(f"   - [{'x' if idx == i else ' '}] "
                  f"{self.tokenizer.decode(an)}")
        print(console_width * "=")
