import hashlib
import os
import shutil
from functools import partial
from pathlib import Path
from typing import *

import datasets
import torch
from datasets import load_dataset, DatasetDict
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from rich import print
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, BatchEncoding

HgDataset = Union[Dataset, DatasetDict]


class BaseDataModule(LightningDataModule):
    """
    A base LightningDataModule for the PennTreeBank dataset as example.
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    dset_script_path_or_id = (
        "ptb_text_only"  # HuggingFace dataset id or local path to script
    )
    text_fields = ["sentence"]  # text fields that should be tokenized
    split_ids = [
        datasets.Split.TRAIN,
        datasets.Split.VALIDATION,
        datasets.Split.TEST,
    ]  # split names
    pt_attributes = [
        "input_ids",
        "attention_mask",
    ]  # attributes to be converted into Tensors

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        data_dir: str = "data/",
        train_batch_size: int = 64,
        eval_batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_length: Optional[int] = 512,
        use_subset: bool = False,
        update_cache: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.dset_id = self.generate_id(locals())
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_subset = use_subset
        self.update_cache = update_cache

        self.max_length = max_length
        self.tokenizer = tokenizer

        self.dataset: Optional[HgDataset] = None
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def generate_id(self, _locals: Dict, exclude: Optional[List[str]] = None) -> str:
        exclude = exclude or []
        _exclude = [
            "train_batch_size",
            "eval_batch_size",
            "num_workers",
            "pin_memory",
            "data_dir",
            "__class__",
            "no_cache",
        ] + exclude
        # flatten kwargs
        for k, v in _locals.pop("kwargs").items():
            _locals[k] = v

        repr = ""
        for k, v in _locals.items():
            if k not in _exclude:
                repr += "-" + v.__repr__().replace(" ", "")

        return hashlib.sha224(repr.encode("utf-8")).hexdigest()

    def tokenize_examples(
        self,
        examples: Dict[str, List[Any]],
        *,
        tokenizer: PreTrainedTokenizerFast,
        max_length: Optional[int],
    ) -> Union[Dict, BatchEncoding]:
        """Tokenize a batch of examples and truncate if `max_length` is provided.
        The input format is:
        examples = {
            attribute_name: list of attribute values
        }
        """
        return tokenizer(
            *(examples[field] for field in self.text_fields),
            max_length=max_length,
            truncation=max_length is not None,
        )

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        self.load_base_dataset()

    def load_base_dataset(self) -> DatasetDict:
        """Load the base HuggingFace dataset."""
        return load_dataset(self.dset_script_path_or_id, cache_dir=self.data_dir)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if self.update_cache or not self.get_cache_path().exists():
            self.dataset: HgDataset = self.load_base_dataset()
            self.dataset = self.filter_dataset(self.dataset)
            if self.use_subset:
                self.dataset = self.take_subset(self.dataset)

            self.dataset = self.preprocess_dataset(self.dataset)

            # save to disk
            self.save_preprocessed_to_disk()

        else:
            self.load_preprocessed_from_disk()

        # assign splits
        self.data_train = self.dataset[self.split_ids[0]]
        self.data_val = self.dataset[self.split_ids[1]]
        self.data_test = self.dataset[self.split_ids[2]]

        self.pprint()

        self.display_sample()

    def take_subset(self, dataset: HgDataset) -> HgDataset:
        """Take a subset of the dataset and return."""
        if isinstance(dataset, DatasetDict):
            return DatasetDict(
                {
                    k: dset.select(range(n))
                    for n, (k, dset) in zip([100, 10, 10], self.dataset.items())
                }
            )
        elif isinstance(dataset, Dataset):
            return dataset.select(range(100))
        else:
            raise NotImplementedError

    def load_preprocessed_from_disk(self):
        """Load the preprocessed self.dataset from disk."""
        self.dataset = DatasetDict().load_from_disk(str(self.get_cache_path()))

    def save_preprocessed_to_disk(self):
        """Save the preprocessed self.dataset to disk."""
        # todo: save to disk should be done automatically through the orriginal Dataset object. Nonetheless, the encode function appears not to be serializable, hence the fingerprint cannot be computed todo: and the encode function cannot be pickled. check if the future version solve the issue and remove this manual save/load operrations"""
        cache_path = self.get_cache_path()
        cache_path.parent.mkdir(exist_ok=True)
        self.dataset.save_to_disk(str(cache_path))

    def preprocess_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply processing steps to the dataset. Tokenization and formatting as PyTorch tensors"""
        fn = partial(
            self.tokenize_examples, tokenizer=self.tokenizer, max_length=self.max_length
        )
        dataset = dataset.map(fn, batched=True)
        dataset.set_format(type="torch", columns=self.pt_attributes)
        return dataset

    def filter_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply filter operation to the dataset and return"""
        return dataset

    def get_cache_path(self) -> Path:
        """Get the path to the cache directory. This cache directory is used for manual saving of the dataset."""
        # TODO: solve issues that will arise due to versioning. Solve by avoiding using custom save_to_disk method.
        fn = f"{type(self).__name__}-{self.dset_script_path_or_id.replace('/', '_')}-{self.dset_id}"
        dir = os.path.join(self.data_dir, "cache")
        return Path(dir) / fn

    def pprint(self):
        """Pretty print the dtaset"""
        print(
            f">> Dataset: [use_subset={self.use_subset}]: \n"
            f"{self.data_train}\n"
            f"{self.data_val}\n"
            f"{self.data_test}\n"
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch: Any) -> Union[BatchEncoding, Dict[str, torch.Tensor]]:
        """The function that is used to merge examples into a batch.
        Concatenating sequences with different length requires padding them."""
        return self.tokenizer.pad(batch)

    @rank_zero_only
    def display_sample(self):
        """Sample a batch and pretty print it."""
        batch = next(iter(self.train_dataloader()))
        console_width, _ = shutil.get_terminal_size()
        print(console_width * "=")
        print("=== Training Batch ===")
        print(console_width * "-")
        for k, v in batch.items():
            print(f"   - {k}: {v.shape} <{v.dtype}>")
        print(console_width * "=")
        self.display_one_sample({k: v[0] for k, v in batch.items()})

    def display_one_sample(self, example: Dict[str, torch.Tensor]):
        """Decode and print one example from the batch"""
        console_width, _ = shutil.get_terminal_size()
        print("=== Sample ===")
        print(console_width * "-")
        print(self.tokenizer.decode(example["input_ids"], skip_special_tokens=True))
        print(console_width * "=")
