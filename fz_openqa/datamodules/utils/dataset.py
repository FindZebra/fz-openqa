from typing import Dict
from typing import List
from typing import Optional

import rich
from datasets import Dataset
from datasets import DatasetDict

from .typing import HfDataset
from fz_openqa.utils.pretty import get_separator


def get_column_names(dataset: HfDataset) -> List[str]:
    if isinstance(dataset, DatasetDict):
        return list(
            set.union(*(set(d.column_names) for d in dataset.values()))
        )
    else:
        return dataset.column_names


def take_subset(dataset: HfDataset, subset_size: List[int]) -> HfDataset:
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


def format_size_difference(
    original_size: Dict[str, int], new_dataset: DatasetDict
) -> str:
    # store the previous split sizes
    prev_lengths = {k: v for k, v in original_size.items()}
    new_lengths = {k: len(v) for k, v in new_dataset.items()}
    u = "Dataset size after filtering ("
    for key in new_lengths.keys():
        ratio = new_lengths[key] / prev_lengths[key]
        u += f"{key}: {new_lengths[key]} ({100 * ratio:.0f}%), "
    return u + ")"


def filter_questions_by_pos_docs(
    row, *, n_documents: int, max_pos_docs: Optional[int]
):
    """
    This function checks if a given row can should be filtered out.
    It will be filtered out if
        1. There are no positive document.
        2. There are not enough negative documents to
           select `n_documents` with at max. `max_pos_docs` positive docs.
    """
    total = len(row["document.match_score"])
    n_positive = sum([int(s > 0) for s in row["document.match_score"]])
    if max_pos_docs is None:
        return n_positive > 0

    n_negatives = total - n_positive
    n_candidates = min(n_positive, max_pos_docs) + n_negatives
    return n_positive > 0 and n_candidates >= n_documents
