from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split
from omegaconf import DictConfig

from ...utils.fingerprint import get_fingerprint
from .typing import HfDataset


def take_subset(
    dataset: HfDataset, subset_size: float | int | Dict[Split, float] | Dict[Split, int]
) -> HfDataset:
    """Take a subset of the dataset and return."""
    if isinstance(dataset, Dataset):
        if not isinstance(subset_size, (float, int)):
            raise ValueError(f"subset_size must be a float or int, got {type(subset_size)}")

        # take a fraction of the dataset
        if isinstance(subset_size, float) and 0 <= subset_size <= 1:
            subset_size = int(subset_size * len(dataset))

        # select subset and return
        rgn = np.random.RandomState(0)
        indices = rgn.choice(len(dataset), subset_size, replace=False)
        return dataset.select(indices)

    elif isinstance(dataset, DatasetDict):
        if isinstance(subset_size, (float, int)):
            subset_size = {split: subset_size for split in dataset.keys()}

        elif not isinstance(subset_size, (DatasetDict, DictConfig)):
            raise TypeError(
                f"subset_size must be a float, int or dict, " f"got {type(subset_size)}"
            )
        return DatasetDict(
            {split: take_subset(ds, subset_size[split]) for split, ds in dataset.items()}
        )
    else:
        raise TypeError(f"dataset must be a Dataset or DatasetDict, got {type(dataset)}")


def format_size_difference(original_size: Dict[str, int], new_dataset: DatasetDict) -> str:
    # store the previous split sizes
    prev_lengths = {k: v for k, v in original_size.items()}
    new_lengths = {k: len(v) for k, v in new_dataset.items()}
    u = "Dataset size after filtering ("
    for key in new_lengths.keys():
        ratio = new_lengths[key] / prev_lengths[key]
        u += f"{key}: {new_lengths[key]} ({100 * ratio:.0f}%), "
    return u + ")"


def filter_questions_by_pos_docs(row, *, level: int = 0, **kwargs):
    if level == 0:
        return filter_questions_by_pos_docs_flat(row, **kwargs)
    elif level == 1:
        return filter_concatenated_questions_by_pos_docs(row, **kwargs)
    else:
        raise ValueError(f"level must be 0 or 1, got {level}")


def filter_questions_by_pos_docs_flat(
    row,
    *,
    n_documents: Union[int, Dict],
    max_pos_docs: Optional[int],
    split: Optional[Split],
):
    """
    This function checks if a given row can should be filtered out.
    It will be filtered out if
        1. There are no positive document.
        2. There are not enough negative documents to
           select `n_documents` with at max. `max_pos_docs` positive docs.
    """
    # get the number of documents corresponding to this split
    if isinstance(n_documents, (dict, DictConfig)):
        n_documents = n_documents[split]

    # get the number of positive documents
    total = len(row["document.match_score"])
    n_positive = sum_of_positives(row["document.match_score"])
    if max_pos_docs is None:
        return n_positive > 0

    # check if there are enough positive documents to fullfill the `max_pos_docs` requirements
    n_negatives = total - n_positive
    n_candidates = min(n_positive, max_pos_docs) + n_negatives
    return n_positive > 0 and n_candidates >= n_documents


def sum_of_positives(values: List) -> int:
    return sum([int(s > 0) for s in values])


def filter_concatenated_questions_by_pos_docs(
    row,
    *,
    n_documents: Union[int, Dict],
    max_pos_docs: Optional[int],
    split: Optional[Split],
):
    """
    This function checks if a given row can should be filtered out.
    It will be filtered out if
        1. There are no positive document for the gold question.
        2. There are not enough negative documents to
           select `n_documents` with at max. `max_pos_docs` positive docs for all options.
    """

    # get the number of documents corresponding to this split
    if isinstance(n_documents, (dict, DictConfig)):
        n_documents = n_documents[split]

    # get the target document and match scores
    target = int(row["answer.target"])
    match_scores_gold = row["document.match_score"][target]

    # get the number of positive documents
    total = len(match_scores_gold)
    n_gold_positive = sum([int(s > 0) for s in match_scores_gold])
    if max_pos_docs is None:
        return n_gold_positive > 0

    # check if there are enough positive documents to fullfill the `max_pos_docs` requirements
    n_positives = [sum_of_positives(opt_docs) for opt_docs in row["document.match_score"]]
    n_negatives = [total - n for n in n_positives]
    n_candidates = [min(npos, max_pos_docs) + nneg for npos, nneg in zip(n_positives, n_negatives)]
    return n_gold_positive > 0 and all(nc >= n_documents for nc in n_candidates)


def get_column_names(dataset: HfDataset) -> List[str]:
    if isinstance(dataset, DatasetDict):
        names = [c for dset in dataset.values() for c in dset.column_names]
        return list(set(names))
    elif isinstance(dataset, Dataset):
        return dataset.column_names
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def remove_columns_with_fingerprint(dataset: Dataset, columns: List[str]) -> Dataset:
    """Remove columns and set a fingerprint deterministically"""
    columns = list(sorted(columns))
    args = {"dataset": dataset._fingerprint, "columns": columns}
    new_fingerprint = get_fingerprint(args)
    return dataset.remove_columns(columns, new_fingerprint=new_fingerprint)


def keep_only_columns(dataset: HfDataset, columns: Optional[List[str]]) -> HfDataset:
    """Keep only the given columns and set a fingerprint deterministically"""
    if columns is None:
        return dataset
    else:
        cols = [c for c in get_column_names(dataset) if c not in columns]
        if isinstance(dataset, Dataset):
            return remove_columns_with_fingerprint(dataset, cols)
        elif isinstance(dataset, DatasetDict):
            return DatasetDict(
                {k: remove_columns_with_fingerprint(v, cols) for k, v in dataset.items()}
            )
