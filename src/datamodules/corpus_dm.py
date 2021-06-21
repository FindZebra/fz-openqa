import os
import re
import shutil
from functools import partial
from typing import *

import datasets
import numpy as np
import rich
import torch
from datasets import load_dataset, DatasetDict
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import Dataset
from transformers import BatchEncoding

from .base_dm import BaseDataModule
from .datasets import corpus
from .utils import gen_passages

HgDataset = Union[Dataset, DatasetDict]

TXT_PATTERN = r"^.*\.txt$"

from src.tokenizers.static import DOC_TOKEN


class CorpusDataModule(BaseDataModule):
    """
    A Corpus data for handling large-scale text datasets. Corpus features the following:
    * efficient caching for handling web-scale datasets (handled by `datasets`)
    * extraction of passages given a fixed length and stride
    * indexing passages
    * querying passages
    """

    dset_script_path_or_id = (
        corpus.__file__  # HuggingFace dataset id or local path to script
    )
    text_fields = ["text"]  # text fields that should be tokenized
    split_ids = [
        datasets.Split.TRAIN,
        datasets.Split.VALIDATION,
        datasets.Split.TEST,
    ]  # split names
    pt_attributes = [
        "idx",
        "passage_idx",
        "input_ids",
        "attention_mask",
    ]  # attributes to be converted into Tensors

    def __init__(
            self,
            *args,
            input_dir: str,
            passage_length: int = 200,
            passage_stride: int = 200,
            max_length=None,
            **kwargs,
    ):
        super().__init__(*args, max_length=max_length, **kwargs)
        assert self.max_length is None, f"`max_length` is not a valid argument for this dataset " \
                                        f"and should be left to None. " \
                                        f"Use the argument `passage_length` instead."
        self.input_dir = input_dir
        self.passage_length = passage_length
        self.passage_stride = passage_stride

    def load_base_dataset(self) -> DatasetDict:
        """Load the base HuggingFace dataset."""
        input_files = [os.path.join(self.input_dir, p) for p in os.listdir(self.input_dir) if
                       re.findall(TXT_PATTERN, p)]
        return load_dataset(self.dset_script_path_or_id, cache_dir=self.data_dir, data_files=input_files)

    @staticmethod
    def add_idx(example: Dict[str, Any], idx: int):
        example['idx'] = idx
        return example

    def preprocess_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply processing steps to the dataset. Tokenization and formatting as PyTorch tensors"""

        # add index column
        dataset = dataset.map(self.add_idx, batched=False, with_indices=True)

        # tokenize the dataset
        fn = partial(
            self.tokenize_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        dataset = dataset.map(fn, batched=True)

        # generate passages of equal size
        doc_token_id = self.tokenizer.get_vocab()[DOC_TOKEN]
        gen_passages = partial(
            self.generate_passages,
            size=self.passage_length,
            stride=self.passage_stride,
            start_tokens=[self.tokenizer.cls_token_id, doc_token_id],
            end_tokens=[self.tokenizer.sep_token_id],
            pad_token_id=self.tokenizer.pad_token_id,
            verbose=self.verbose,
        )
        dataset = dataset.remove_columns(["text"])
        dataset = dataset.map(gen_passages, batched=True)
        dataset.set_format(type="torch", columns=self.pt_attributes)
        return dataset

    @staticmethod
    def generate_passages(
            examples: Dict[str, List[Any]],
            *,
            size: int,
            stride: int,
            start_tokens: List[int],
            end_tokens: List[int],
            pad_token_id: int,
            verbose: bool = True,
    ) -> Dict[str, List]:
        if verbose:
            lens = list(map(len, examples["input_ids"]))
            rich.print(
                f">> @CorpusDataModule.generate_windows: Number of tokens per documents: "
                f"mean={np.mean(lens):.1f}, std={np.std(lens):.1f} [{min(lens)}-{max(lens)}]"
            )

        # extend idxs, the attention mask and compute the passage masks and add passage_ids
        base_args = {"size": size, "stride": stride}
        args = {
            "pad_token": 0,
            "start_tokens": [0 for _ in start_tokens],
            "end_tokens": [0 for _ in start_tokens],
        }
        idxs, w_idxs, attention_mask, window_mask = zip(
            *[
                (i, w_i, w, m)
                for i, ex in zip(examples["idx"], examples["attention_mask"])
                for w_i, (w, m) in enumerate(gen_passages(ex, **args, **base_args))
            ]
        )
        output = {
            "idx": list(idxs),
            "passage_idx": list(w_idxs),
            "attention_mask": list(attention_mask),
            "passage_mask": list(window_mask),
        }

        # update "'input_ids' (potentially list other attributes here)
        args_dict = {
            "input_ids": {
                "pad_token": pad_token_id,
                "return_mask": False,
                "start_tokens": start_tokens,
                "end_tokens": end_tokens,
            },
        }
        output.update(
            {
                k: [
                    w
                    for ex in examples[k]
                    for w in gen_passages(ex, **base_args, **args)
                ]
                for k, args in args_dict.items()
            }
        )

        return output

    @rank_zero_only
    def display_sample(self):
        """Sample a batch and pretty print it."""
        batch = next(iter(self.train_dataloader()))
        console_width, _ = shutil.get_terminal_size()
        print(console_width * "=")
        print("=== Training Batch ===")
        print(console_width * "-")
        for k, v in batch.items():
            rich.print(f"   - {k}: {v.shape} <{v.dtype}>")
        print(console_width * "=")
        for i in range(3):
            self.display_one_sample({k: v[i] for k, v in batch.items()})

    def display_one_sample(self, example: Dict[str, torch.Tensor]):
        """Decode and print one example from the batch"""
        console_width, _ = shutil.get_terminal_size()
        print("=== Sample ===")
        print(console_width * "-")
        print(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False))
        print(console_width * "=")

    def collate_fn(self, batch: Any) -> Union[BatchEncoding, Dict[str, torch.Tensor]]:
        """The function that is used to merge examples into a batch.
        Concatenating sequences with different length requires padding them."""
        return self.tokenizer.pad(batch)
