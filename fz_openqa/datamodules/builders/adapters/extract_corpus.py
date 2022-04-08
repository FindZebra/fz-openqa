from collections import OrderedDict
from typing import List

from datasets import concatenate_datasets
from datasets import DatasetDict


def extract_corpus(dataset: DatasetDict, key: str, **kwargs) -> (DatasetDict, List):
    """Extract corpus from a QA dataset (e.g. `SQUAD`)."""
    # take the columns corresponding to `key` in each dataset
    values = {s: d[key] for s, d in dataset.items()}
    if len(set.intersection(*map(set, values.values()))) > 0:
        raise ValueError(f"{key} is not unique: " f"{set.intersection(*map(set, values.values()))}")

    # get a lookup table for each split, from unique value to original index
    index_lookup_ = {s: OrderedDict({x: i for i, x in enumerate(xs)}) for s, xs in values.items()}

    # extract the corpus (unique rows)
    unique_indices = {s: list(xs.values()) for s, xs in index_lookup_.items()}
    corpus = {s: d.select(unique_indices[s]) for s, d in dataset.items()}

    # merge the splits into a single dataset
    corpus = concatenate_datasets(list(corpus.values()))
    keys = corpus[key]
    return corpus, keys
