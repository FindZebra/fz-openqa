from typing import Dict
from typing import List
from typing import Optional

import rich
from datasets import Dataset
from datasets import DatasetDict

from .typing import HgDataset
from fz_openqa.utils.pretty import get_separator


def get_column_names(dataset: HgDataset) -> List[str]:
    if isinstance(dataset, DatasetDict):
        return list(
            set.union(*(set(d.column_names) for d in dataset.values()))
        )
    else:
        return dataset.column_names


def take_subset(dataset: HgDataset, subset_size: List[int]) -> HgDataset:
    """Take a subset of the dataset and return."""
    if isinstance(dataset, DatasetDict):
        return DatasetDict(
            {
                k: dset.select(range(n))
                for n, (k, dset) in zip(subset_size, dataset.items())
            }
        )
    elif isinstance(dataset, Dataset):
        size = next(iter(subset_size))
        return dataset.select(range(size))
    else:
        raise NotImplementedError


def print_size_difference(
    original_size: Dict[str, int], new_dataset: DatasetDict
):
    # store the previous split sizes
    prev_lengths = {k: v for k, v in original_size.items()}
    new_lengths = {k: len(v) for k, v in new_dataset.items()}
    print(get_separator())
    rich.print("> New dataset size:")
    for key in new_lengths.keys():
        ratio = new_lengths[key] / prev_lengths[key]
        rich.print(f">  - {key}: {new_lengths[key]} ({100 * ratio:.2f}%)")
    print(get_separator())


def filter_questions_by_pos_docs(row, *, max_pos_docs: Optional[int]):
    max_pos_docs = max_pos_docs or 1e20
    n = sum([int(s > 0) for s in row["document.match_score"]])
    return n > 0 and n <= max_pos_docs
