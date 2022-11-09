from __future__ import annotations

import abc
from typing import List
from typing import Optional

from datasets import Dataset
from datasets import DatasetDict
from loguru import logger
from warp_pipes import HfDataset


class DatasetPreprocessing:
    def __init__(
        self, only_train_set: bool = False, dset_name: Optional[str | List[str]] = None, **kwargs
    ):
        if isinstance(dset_name, str):
            dset_name = [dset_name]
        self.dset_name = dset_name
        if only_train_set:
            self.allowed_splits = {"train"}
        else:
            self.allowed_splits = None

    def __call__(self, dataset: HfDataset, *, dset_name: str, **kwargs) -> HfDataset:
        if self.dset_name is not None:
            if dset_name not in self.dset_name:
                logger.info(
                    f"Skipping {dset_name} for {type(self).__name__} "
                    f"(accepted names: {self.dset_name})"
                )
                return dataset

        if isinstance(dataset, Dataset):
            return self.preprocess(dataset, **kwargs)
        elif isinstance(dataset, DatasetDict):
            return DatasetDict(
                {split: self.maybe_preprocess(split, v, **kwargs) for split, v in dataset.items()}
            )
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    @abc.abstractmethod
    def preprocess(self, dataset: Dataset, dset_name: Optional[str] = None, **kwargs) -> Dataset:
        ...

    def maybe_preprocess(self, split: str, dataset: Dataset, **kwargs) -> Dataset:
        if self.allowed_splits is None or split in self.allowed_splits:
            return self.preprocess(dataset, **kwargs)
        else:
            return dataset
