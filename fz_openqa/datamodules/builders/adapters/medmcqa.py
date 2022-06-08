from typing import Dict

import datasets
from datasets import Dataset
from datasets import DatasetDict

from fz_openqa.datamodules.builders.adapters.base import DatasetAdapter


class NestColumns:
    def __init__(self, input_columns, output_column):
        self.input_columns = input_columns
        self.output_column = output_column

    def __call__(self, row: Dict) -> Dict:
        output = [row[c] for c in self.input_columns]
        return {self.output_column: output}


class MedMCQAAdapter(DatasetAdapter):
    def __call__(self, dataset: DatasetDict, **kwargs) -> (DatasetDict, Dataset):
        return DatasetDict({split: self.format(dset) for split, dset in dataset.items()}), None

    def format(self, dataset: Dataset, **kwargs) -> Dataset:
        # nest the answer options
        dataset = dataset.map(
            NestColumns(["opa", "opb", "opc", "opd"], "answer.text"),
            desc="Cleanup answer options",
            **kwargs
        )
        dataset = dataset.rename_columns(
            {
                "question": "question.text",
                "cop": "answer.target",
                "exp": "reasoning",
            }
        )

        # cast answer.target
        features = dataset.features.copy()
        features["answer.target"] = datasets.Value("int32")
        dataset = dataset.cast(features=features)

        # drop columns
        dataset = dataset.remove_columns(["opa", "opb", "opc", "opd"])
        return dataset
