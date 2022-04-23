from typing import Optional

from datasets import Dataset
from datasets import DatasetDict


class DatasetAdapter:
    def __call__(self, dataset: DatasetDict, **kwargs) -> (DatasetDict, Optional[Dataset]):
        ...
