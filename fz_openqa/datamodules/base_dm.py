from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import rich
import torch
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
from datasets import Split
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from fz_openqa.datamodules.pipes import Lambda
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import TokenizerPipe
from fz_openqa.datamodules.utils import HgDataset
from fz_openqa.datamodules.utils import take_subset
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pprint_batch
from fz_openqa.utils.pretty import pretty_decode


class BaseDataModule(LightningDataModule):
    """
    A task agnostic base datamodule. This implements and showcase the
    basic functionalities of a text DataModule.

    Implementing a sub-class of a `BaseDataModule` mostly requires overriding the
    `.preprocessing()` and `.collate_fn()` methods.

    <original documentation>
    A DataModule implements 5 key methods:
        - prepare_data (things to do on every noe)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    # HuggingFace dataset id or local path to script
    dset_script_path_or_id = "ptb_text_only"

    # text fields from the raw datasets that should be tokenized and concatenated
    text_fields = ["sentence"]

    # name of the attributes that will be converted to
    # tensors in the preprocessing function
    pt_attributes = ["input_ids", "attention_mask"]

    # number of data points per subset train/val/test
    subset_size = [100, 10, 10]

    # the attribute used to store the dataset
    dataset: HgDataset = None

    # define the operator that allows converting a sequence of
    # examples into a Batch
    collate_pipe: Pipe = None

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        cache_dir: str = "cache/",
        train_batch_size: int = 64,
        eval_batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        max_length: Optional[int] = 512,
        use_subset: bool = False,
        num_proc: int = 1,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = cache_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.use_subset = use_subset
        self.num_proc = num_proc
        self.verbose = verbose

        # tokenizer and dataset
        self.max_length = max_length
        self.tokenizer = tokenizer

    def load_base_dataset(self) -> DatasetDict:
        """Load the base HuggingFace dataset."""
        return load_dataset(
            self.dset_script_path_or_id, cache_dir=self.data_dir
        )

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        self.load_base_dataset()

    def setup(self, stage: Optional[str] = None):
        """
        Load data and preprocess the data.
        1. Store all data into the attribute `self.dataset` using `self.preprocess_dataset`
        2. Build the operator to collate examples into a batch (`self.collate_pipe`).
        """

        # preprocess
        self.dataset: HgDataset = self.load_base_dataset()
        self.dataset = self.filter_dataset(self.dataset)
        if self.use_subset:
            self.dataset = take_subset(self.dataset, self.subset_size)
        self.dataset = self.preprocess_dataset(self.dataset)

        # define the collate operator
        self.collate_pipe = self.get_collate_pipe()

    def preprocess_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply processing steps to the dataset.
        Tokenization and formatting as PyTorch tensors"""
        pipe = TokenizerPipe(
            self.tokenizer,
            max_length=self.max_length,
            fields=self.text_fields,
            return_token_type_ids=False,
            add_special_tokens=False,
            return_offsets_mapping=False,
        )

        dataset = dataset.map(
            pipe,
            batched=True,
            num_proc=self.num_proc,
            desc="Tokenizing",
            remove_columns=self.text_fields,
        )
        dataset.set_format(type="torch", columns=self.pt_attributes)
        return dataset

    def get_collate_pipe(self) -> Pipe:
        return Lambda(lambda examples: self.tokenizer.pad(examples))

    def filter_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply filter operation to the dataset and return"""
        return dataset

    def train_dataloader(self):
        dset = self.get_dataset(Split.TRAIN)

        return DataLoader(
            dataset=dset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def _eval_loader(self, split):
        dset = self.get_dataset(split)

        return DataLoader(
            dataset=dset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return self._eval_loader(Split.VALIDATION)

    def test_dataloader(self):
        return self._eval_loader(Split.TEST)

    def get_dataset(
        self, split: Union[str, Split], dataset: Optional[HgDataset] = None
    ) -> Dataset:
        """Return the dataset corresponding to the split,
        or the dataset iteself if there is no split."""
        dataset = dataset or self.dataset
        if isinstance(dataset, Dataset):
            return dataset
        elif isinstance(dataset, DatasetDict):
            return dataset[split]
        else:
            raise TypeError(f"Unknown dataset type <{type(dataset).__name__}>")

    def collate_fn(self, examples: List[Batch]) -> Batch:
        """The function that is used to merge examples into a batch.
        Concatenating sequences with different length requires padding them."""
        return self.collate_pipe(examples)

    def pprint(self):
        """Pretty print the dtaset"""
        rich.print(
            f">> Dataset: [use_subset={self.use_subset}]: \n" f"{self.dataset}"
        )

    @rank_zero_only
    def display_sample(self):
        """Sample a batch and pretty print it."""
        batch = next(iter(self.train_dataloader()))
        eval_batch = next(iter(self.val_dataloader()))
        print(get_separator("="))
        print("=== Training Batch ===")
        print(get_separator())
        pprint_batch(batch)
        print(get_separator())
        print("=== Validation Batch ===")
        print(get_separator())
        pprint_batch(eval_batch)
        print(get_separator())
        print("=== Training Example ===")
        print(get_separator())
        self.display_one_sample({k: v[0] for k, v in batch.items()})
        print(get_separator("="))

    def display_one_sample(self, example: Dict[str, torch.Tensor]):
        """Decode and print one example from the batch"""
        rich.print(
            pretty_decode(
                example["input_ids"],
                tokenizer=self.tokenizer,
                skip_special_tokens=True,
            )
        )
