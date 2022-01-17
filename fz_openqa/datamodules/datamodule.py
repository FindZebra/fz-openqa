import logging
from functools import partial
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import rich
import torch
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from fz_openqa.datamodules.builders.base import DatasetBuilder
from fz_openqa.datamodules.pipes import Partial
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.utils.typing import HfDataset
from fz_openqa.utils import maybe_instantiate
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import infer_batch_size
from fz_openqa.utils.pretty import get_separator
from fz_openqa.utils.pretty import pprint_batch

logger = logging.getLogger(__name__)


class DataModule(LightningDataModule):
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

    # the attribute used to store the dataset
    dataset: DatasetDict = None

    # define the operator that allows converting a sequence of
    # examples into a Batch
    collate_pipe: Pipe = None

    def __init__(
        self,
        *,
        builder: Union[DatasetBuilder, DictConfig],
        train_batch_size: int = 64,
        eval_batch_size: int = 128,
        num_workers: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        drop_last: bool = False,
        **kwargs,
    ):
        super().__init__()

        # builder used to generate the dataset
        self.builder = maybe_instantiate(builder)

        # data loader parameters
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last

    def prepare_data(self, **kwargs):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        logger.info(f"Preparing data with <{self.builder.__class__.__name__}>")
        self.builder(**kwargs)

    def setup(self, stage: Optional[str] = None, **kwargs):
        """
        Load data and preprocess the data.
        1. Store all data into the attribute `self.dataset` using `self.preprocess_dataset`
        2. Build the operator to collate examples into a batch (`self.collate_pipe`).
        """
        logger.info(f"Setting up with <{self.builder.__class__.__name__}>")
        self.update_dataset(stage=stage, **kwargs)

    def update_dataset(self, stage: Optional[str] = None, **kwargs):
        """
        Load data and preprocess the data.
        1. Store all data into the attribute `self.dataset` using `self.preprocess_dataset`
        2. Build the operator to collate examples into a batch (`self.collate_pipe`).
        """
        logger.info(f"Updating dataset with <{self.builder.__class__.__name__}>")
        try:
            del self.dataset
        except AttributeError:
            pass
        self.dataset = self.builder(**kwargs)

        # define the collate operator
        try:
            del self.collate_pipe
        except AttributeError:
            pass
        self.collate_pipe = self.builder.get_collate_pipe()

    def train_dataloader(self, *, shuffle: bool = True):
        collate_fn = self._get_collate_fn(split=Split.TRAIN)

        return DataLoader(
            dataset=self.get_dataset(Split.TRAIN),
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

    def _eval_loader(self, split, *, shuffle: bool = False):
        collate_fn = self._get_collate_fn(split=split)

        return DataLoader(
            dataset=self.get_dataset(split),
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

    def _get_collate_fn(self, split: Split):
        collate_fn = self.collate_pipe

        # add the split info to the collate_fn
        if isinstance(collate_fn, Pipe):
            collate_fn = partial(collate_fn, split=split)

        return collate_fn

    def val_dataloader(self, **kwargs):
        return self._eval_loader(Split.VALIDATION, **kwargs)

    def test_dataloader(self, **kwargs):
        return self._eval_loader(Split.TEST, **kwargs)

    def get_dataset(
        self, split: Union[str, Split], dataset: Optional[HfDataset] = None
    ) -> Union[TorchDataset, Dataset]:
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

    def __repr__(self):
        return f"{self.__class__.__name__}(\nbuilder={self.builder}\n)"

    @rank_zero_only
    def display_samples(self, n_samples: int = 1):
        """Sample a batch and pretty print it."""
        batch = next(iter(self.train_dataloader()))
        print(get_separator("="))
        print("=== Batch ===")
        print(get_separator())
        pprint_batch(batch)
        print(get_separator())
        print("=== example ===")
        try:
            for i in range(min(n_samples, infer_batch_size(batch))):
                print(get_separator())
                self.display_one_sample({k: v[i] for k, v in batch.items()})
        except Exception as e:
            logger.exception(e)
        print(get_separator("="))

    def display_one_sample(self, example: Dict[str, torch.Tensor]):
        """Decode and print one example from the batch"""
        rich.print(self.builder.format_row(example))
