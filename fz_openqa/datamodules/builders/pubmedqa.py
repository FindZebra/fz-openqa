from typing import Optional

import datasets
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from datasets.splits import Split

from .hf_dataset import HfDatasetBuilder
from fz_openqa.datamodules.utils.typing import HfDataset


class PubMedQaBuilder(HfDatasetBuilder):
    # HuggingFace dataset id
    dset_script_path_or_id = "pqa_labeled"

    pt_attributes = [
        "question.input_ids",
        "question.attention_mask",
        "pubid",
    ]

    def __init__(self, *args, split: Split, **kwargs):
        super(PubMedQaBuilder, self).__init__(*args, **kwargs)
        self.split = str(split)

    def load_base_dataset(self) -> DatasetDict:
        dataset = load_dataset("pubmed_qa", self.dset_script_path_or_id)  # todo: make more dynamic
        dataset = datasets.Dataset.train_test_split(
            dataset["train"], train_ratio=0.8
        )  # todo: use input parameter split
        dataset_eval = datasets.Dataset.train_test_split(dataset["test"], train_ratio=0.5)
        dataset["validation"] = dataset_eval["train"]
        dataset["test"] = dataset_eval["test"]

        return dataset
