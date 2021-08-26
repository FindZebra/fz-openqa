from typing import Sized

from torch.utils.data import Dataset as TorchDataset


class Sampler(TorchDataset):
    """A basic dataset sampler. A sampler wraps an existing dataset and modifies the methods
    __getitem__ and __len__."""

    def __init__(self, *, dataset: Sized):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
