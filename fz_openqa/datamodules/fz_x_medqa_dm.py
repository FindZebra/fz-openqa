import random
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import rich
from datasets import concatenate_datasets
from datasets import Dataset
from datasets import DatasetDict
from datasets import Split

from .corpus_dm import CorpusDataModule
from .datasets import fz_x_medqa
from .pipes import Parallel
from .qa_dm import QaDatamodule
from .qa_dm import set_example_idx
from .utils import HgDataset

PT_SIMPLE_ATTRIBUTES = [
    "answer.target",
    "answer.n_options",
    "document.rank",
    "document.is_positive",
    "question.idx",
    "idx",
]


def get_columns(dataset: HgDataset) -> List[str]:
    if isinstance(dataset, DatasetDict):
        return list(
            set.intersection(
                *(set(dset.column_names) for dset in dataset.values())
            )
        )

    else:
        return dataset.column_names


def isolate_corpus(
    dataset: HgDataset,
    *,
    index_column: str = "question.idx",
    corpus_key: str = "document",
) -> Tuple[HgDataset, Dataset]:

    # get a clean version of the dataset without the `corpus_key`columnns
    dset_columns = [
        key for key in get_columns(dataset) if f"{corpus_key}." in key
    ]
    filtered_dataset = dataset.remove_columns(dset_columns)

    # get the columns corresponding to the corpus + the index column
    corpus_columns = [
        key
        for key in get_columns(dataset)
        if (f"{corpus_key}." not in key and key != index_column)
    ]
    rich.print(f">> corpus_columns: {corpus_columns}")
    corpus = dataset.remove_columns(corpus_columns)

    # cast the corpus to Dataset
    if isinstance(corpus, DatasetDict):
        corpus = concatenate_datasets(list(corpus.values()))

    return filtered_dataset, corpus


class FZxMedQADataModule(QaDatamodule):
    """A PyTorch Lightning DataModule wrapping the FZxMedQA dataset."""

    # HuggingFace dataset id or local path to script
    dset_script_path_or_id = fz_x_medqa.__file__

    # name of the attributes that will be converted to
    # tensors in the preprocessing function
    pt_attributes = [
        "question.input_ids",
        "question.attention_mask",
        "question.idx",
        "answer.input_ids",
        "answer.attention_mask",
        "answer.target",
        "document.idx",
        "document.input_ids",
        "document.attention_mask",
        "document.is_positive",
    ]

    def __init__(self, *, filter_gold: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.filter_gold = filter_gold

    def preprocess_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply processing steps to the dataset.
        Tokenization and formatting as PyTorch tensors"""

        dataset, corpus = isolate_corpus(
            dataset, corpus_key="document", index_column="question.idx"
        )

        # build a corpus Datamodule form the datasets.dataset
        self.corpus = CorpusDataModule.from_dataset(corpus)

        # preprocess the dataset as a QaDatamodule
        return super(FZxMedQADataModule, self).preprocess_dataset(dataset)

    def filter_dataset(self, dataset: HgDataset) -> HgDataset:
        """Apply filtering operations"""
        if self.filter_gold:
            dataset = dataset.filter(
                lambda x: x["document.rank"] == 0 and x["document.is_positive"]
            )

        return dataset

    @staticmethod
    def filter_question_id(ids: List[int], row: Dict[str, Any]) -> bool:
        return row["question.idx"] in ids

    @staticmethod
    def take_subset(dataset: HgDataset) -> HgDataset:
        """Take a subset of the dataset and return."""
        subset_size = {Split.TRAIN: 5, Split.VALIDATION: 2, Split.TEST: 2}
        for key, dset in dataset.items():
            questions_ids = dset["question.idx"]
            selected_ids = random.sample(questions_ids, k=subset_size[key])
            fn = partial(FZxMedQADataModule.filter_question_id, selected_ids)
            dataset[key] = dset.filter(fn)

        return dataset
