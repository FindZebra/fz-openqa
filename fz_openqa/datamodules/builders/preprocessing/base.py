import abc

from datasets import Dataset
from datasets import DatasetDict
from warp_pipes import HfDataset


class DatasetPreprocessing:
    def __init__(self, only_train_set: bool = False):
        if only_train_set:
            self.allowed_splits = {"train"}
        else:
            self.allowed_splits = None

    def __call__(self, dataset: HfDataset, **kwargs) -> HfDataset:
        if isinstance(dataset, Dataset):
            return self.preprocess(dataset, **kwargs)
        elif isinstance(dataset, DatasetDict):
            return DatasetDict(
                {split: self.maybe_preprocess(split, v, **kwargs) for split, v in dataset.items()}
            )
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    @abc.abstractmethod
    def preprocess(self, dataset: Dataset, **kwargs) -> Dataset:
        ...

    def maybe_preprocess(self, split: str, dataset: Dataset, **kwargs) -> Dataset:
        if self.allowed_splits is None or split in self.allowed_splits:
            return self.preprocess(dataset, **kwargs)
        else:
            return dataset
