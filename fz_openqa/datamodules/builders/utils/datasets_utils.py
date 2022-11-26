from typing import List
from typing import Optional

from datasets import Dataset
from datasets import DatasetDict
from warp_pipes import HfDataset
from warp_pipes import infer_batch_shape


def infer_dataset_batch_size(
    dataset: HfDataset, keys: Optional[List[str]] = None, n: int = 10
) -> List[int]:
    assert isinstance(dataset, (Dataset, DatasetDict))
    dset = dataset if isinstance(dataset, Dataset) else next(iter(dataset.values()))
    sample = {k: v for k, v in dset[:n].items() if keys is not None and k in keys}
    batch_size = infer_batch_shape(sample)
    batch_size[0] = -1
    return batch_size


def infer_dataset_nesting_level(*args, **kwargs) -> int:
    shape = infer_dataset_batch_size(*args, **kwargs)
    return len(shape) - 1
