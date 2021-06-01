from typing import *

import torch
from datasets import load_dataset, DatasetDict
from rich import print
from transformers import BatchEncoding

from .base import BaseDatamodule

_SCRIPT_PATH = 'datasets/fz_x_medqa.py'


class FZxMedQA(BaseDatamodule):
    dset_id = "fz_x_medqa"  # HuggingFace dataset id (not used for now)
    split_ids = ["train", "test"]  # split names

    def __init__(self,
                 *,
                 filter_gold: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.filter_gold = True
        if self.use_subset:
            raise NotImplementedError

    def encode(self, examples: Dict[str, Any]) -> Union[Dict, BatchEncoding]:
        # process questions and documents
        q_encodings = self.tokenizer(examples['question'], return_token_type_ids=False)
        d_encodings = self.tokenizer(examples['document'], return_token_type_ids=False)
        output = {'question.text': examples['question'], 'document.text': examples['document']}
        for data, prefix in zip([q_encodings, d_encodings], ['question', 'document']):
            for k, v in data.items():
                output[f"{prefix}.{k}"] = v

        # process answers
        n_choices = len(examples['answer_choices'][0])
        answer_encodings = [self.tokenizer([ans[n] for ans in examples['answer_choices']], return_token_type_ids=False)
                            for n in range(n_choices)]

        for idx, data in enumerate(answer_encodings):
            for k, v in data.items():
                output[f"answer_{idx}.{k}"] = v

        return output

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        self.load_datasets()

    def load_datasets(self, ) -> DatasetDict:
        # TODO: subset

        return DatasetDict(
            {
                split: load_dataset(
                    self.dset_id, cache_dir=self.data_dir, split=f"{slic}"
                )
                for split, slic in zip(
                self.split_ids, ["train[:50%]", "train[50%:]"]
            )
            }
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dsets = self.load_datasets()

        print(100 * "~")
        print(dsets['train'])

        # keep only gold passages
        dsets = dsets.filter(lambda x: x['is_gold'])

        # tokenize and format as PyTorch tensors
        for k in dsets.keys():
            dsets[k] = dsets[k].map(self.encode, batched=True)

        attrs = ['input_ids', 'attention_mask']
        columns = ['question', 'document', 'answer_']
        all_columns = [c for c in dsets.column_names['train'] if
                       (any(a in c for a in attrs) and any(a in c for a in columns))]
        all_columns += ['answer_idx', 'is_gold']
        for k in dsets.keys():
            dsets[k].set_format(type="torch", columns=all_columns, output_all_columns=False)

        # assign splits
        self.data_train = dsets[self.split_ids[0]]
        self.data_val = dsets[self.split_ids[1]]
        self.data_test = dsets[self.split_ids[1]]

        self.pprint()

    def collate_fn(self, batch: Any) -> Union[BatchEncoding, Dict[str, torch.Tensor]]:
        attrs = ['input_ids', 'attention_mask']
        output = {}
        print(100 * "~")
        print(batch[0].keys())

        # answer_idx & is_gold attributes
        output['answer_idx'] = torch.tensor([b['answer_idx'] for b in batch])
        output['is_gold'] = torch.tensor([b['is_gold'] for b in batch])

        # documents and questions
        for key in ['document', 'question']:
            doc_encoding = self.tokenizer.pad(
                [{k.replace(f"{key}.", ''): v for k, v in b.items() if f"{key}." in k} for b in batch])
            for attr, tsr in doc_encoding.items():
                output[f"{key}.{attr}"] = tsr

        # merge answers:
        ans_cols = ['answer_0', 'answer_1', 'answer_2', 'answer_3']
        ans_encoding = self.tokenizer.pad(
            {attr: [b[f"{ans}.{attr}"] for ans in ans_cols for b in batch] for attr in attrs})
        for k, v in ans_encoding.items():
            output[f"answer_choices.{k}"] = v.view(len(ans_cols), len(output['answer_idx']), -1).permute(1, 0,
                                                                                                         2).contiguous()

        return output
    
