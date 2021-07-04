import os
import re
import shutil
from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import datasets
import numpy as np
import rich
import torch
from datasets import DatasetDict
from datasets import load_dataset
from datasets import Split
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
from torch.utils.data import Dataset
from transformers import BatchEncoding

from .base_dm import BaseDataModule
from .datasets import file_corpus
from .datasets import meqa_en_corpus
from .utils import gen_passages
from fz_openqa.tokenizers.static import DOC_TOKEN

HgDataset = Union[Dataset, DatasetDict]

TXT_PATTERN = r"^.*\.txt$"


class CorpusDataModule(BaseDataModule):
    """
    A Corpus data for handling large-scale text datasets. Corpus features the following:
    * efficient caching for handling web-scale datasets (handled by `datasets`)
    * extraction of passages given a fixed length and stride
    * indexing passages
    * querying passages
    """

    dset_script_path_or_id = (
        file_corpus.__file__  # HuggingFace dataset id or local path to script
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
    vectors_id = "vectors"

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
        assert self.max_length is None, (
            "`max_length` is not a valid argument for this dataset "
            "and should be left to None. "
            "Use the argument `passage_length` instead."
        )
        self.input_dir = input_dir
        self.passage_length = passage_length
        self.passage_stride = passage_stride

    def load_base_dataset(self) -> DatasetDict:
        """Load the base HuggingFace dataset."""
        input_files = (
            [
                os.path.join(self.input_dir, p)
                for p in os.listdir(self.input_dir)
                if re.findall(TXT_PATTERN, p)
            ]
            if self.input_dir is not None
            else None
        )
        return load_dataset(
            self.dset_script_path_or_id,
            cache_dir=self.data_dir,
            data_files=input_files,
        )

    @staticmethod
    def add_idx(example: Dict[str, Any], idx: int):
        example["idx"] = idx
        return example

    def preprocess_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply processing steps to the dataset. Tokenization and formatting as PyTorch tensors"""

        # add index column
        dataset = dataset.map(
            self.add_idx, batched=False, with_indices=True, desc="Indexing"
        )

        # tokenize the dataset
        fn = partial(
            self.tokenize_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        dataset = dataset.map(
            fn, batched=True, num_proc=self.num_proc, desc="Tokenizing"
        )

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
        dataset = dataset.map(
            gen_passages,
            batched=True,
            num_proc=self.num_proc,
            desc="Extracting passages",
        )
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
                f"mean={np.mean(lens):.1f}, std={np.std(lens):.1f} [{min(lens)} - {max(lens)}]"
            )

        # extend idxs, the attention mask and compute the passage masks and add passage_ids
        base_args = {"size": size, "stride": stride}
        args = {
            "pad_token": 0,
            "start_tokens": [0 for _ in start_tokens],
            "end_tokens": [0 for _ in end_tokens],
        }
        idxs, w_idxs, attention_mask, window_mask = zip(
            *[
                (i, w_i, w, m)
                for i, ex in zip(examples["idx"], examples["attention_mask"])
                for w_i, (w, m) in enumerate(
                    gen_passages(ex, **args, **base_args)
                )
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
        print("=== Samples ===")
        for i in range(min(3, len(list(batch.values())[0]))):
            self.display_one_sample({k: v[i] for k, v in batch.items()})
        print(console_width * "=")

    def display_one_sample(self, example: Dict[str, torch.Tensor]):
        """Decode and print one example from the batch"""
        console_width, _ = shutil.get_terminal_size()
        decode_kwargs = {"skip_special_tokens": True}
        print(console_width * "-")
        rich.print(
            "(CORPUS) " + self.repr_ex(example, "input_ids", **decode_kwargs)
        )

    def collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> Union[BatchEncoding, Dict[str, torch.Tensor]]:
        """The function that is used to merge examples into a batch.
        Concatenating sequences with different length requires padding them."""
        output = self.tokenizer.pad(batch)

        # infer `max_length`
        tokens = [b["input_ids"] for b in batch]
        pad_tok = self.tokenizer.pad_token_id
        max_length = len(tokens[0]) - min(
            map(lambda x: sum([int(t == pad_tok) for t in x]), tokens)
        )

        def maybe_truncate(x: Any, max_length: int):
            """truncate sequential attributes to `max_length`"""
            if not (isinstance(x, torch.Tensor) and len(x.shape) == 2):
                return x

            return x[:, :max_length]

        return {k: maybe_truncate(v, max_length) for k, v in output.items()}

    @staticmethod
    @torch.no_grad()
    def compute_vectors_batch(
        key: str, model: Callable, batch: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """Compute one batch of vectors"""
        if isinstance(model, torch.nn.Module):
            device = next(iter(model.parameters()))
            batch = {k: v.to(device) for k, v in batch.items()}

        # process and return
        batch[key] = model(batch)
        return batch

    def compute_vectors(self, model: Callable, index: bool = True, **kwargs):
        """Compute the vectors for each passage in the corpus"""
        # todo: distributed version
        self.dataset = self.dataset.map(
            partial(self.compute_vectors_batch, self.vectors_id, model),
            batch_size=self.eval_batch_size,
            num_proc=1,
        )
        if index:
            self.dataset["train"].add_faiss_index(
                column=self.vectors_id, **kwargs
            )

    def query(self, vector: Tensor, k: int = 1):
        """Query the faiss index given a vector query of shape (h,)"""
        # todo: this causes segmentation fault on MacOS, works fine on the cluster
        vector = vector.cpu().numpy()
        return self.dataset["train"].get_nearest_examples(
            self.vectors_id, vector, k=k
        )

    def query_batch(self, vectors: Tensor, k: int = 1):
        """Query the faiss index given a batch of vector queries of shape (bs, h,)"""
        vectors = vectors.cpu().numpy()
        return self.dataset["train"].get_nearest_examples_batch(
            self.vectors_id, vectors, k=k
        )

    def val_dataloader(self):
        return self._eval_loader(
            Split.TRAIN
        )  # the dataset only have one split

    def test_dataloader(self):
        return self._eval_loader(
            Split.TRAIN
        )  # the dataset only have one split


class MedQaEnDataModule(CorpusDataModule):
    dset_script_path_or_id = (
        meqa_en_corpus.__file__  # HuggingFace dataset id or local path to script
    )
