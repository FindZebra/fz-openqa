import re
from typing import Dict

import datasets
from datasets import Dataset
from datasets import DatasetDict

from fz_openqa.datamodules.builders.adapters.base import DatasetAdapter


# https://regex101.com/r/9YNTyr/1
medmcqa_ans_pattern = re.compile(
    (
        r"^((ans|answer)?(\.|:|-)?( *)?(is )?)?"
        r"((\(|\"| |')?[a-d](?!\w))(\)|\"| |')?"
        r"([ ]+i.e.[(,|.)])?( +)?"
    ),
    flags=re.IGNORECASE,
)


class NestColumns:
    def __init__(self, input_columns, output_column):
        self.input_columns = input_columns
        self.output_column = output_column

    def __call__(self, row: Dict) -> Dict:
        output = [row[c] for c in self.input_columns]
        return {self.output_column: output}


class MedMCQACleanupReasoning(object):
    def __init__(self, reasoning_column: str = "exp"):
        self.reasoning_column = reasoning_column

    def __call__(self, row: Dict) -> Dict:
        reasoning = row[self.reasoning_column]
        if reasoning is None:
            cleaned_reasoning = ""
        else:
            cleaned_reasoning = re.sub(medmcqa_ans_pattern, "", reasoning)
            # color = "red" if "ans" in cleaned_reasoning.lower() else "green"
            # rich.print(
            #     f"[gray]>>> {reasoning}\n"
            #     f"[{color}]>> {len(cleaned_reasoning)} "
            #     f">> ({type(cleaned_reasoning)}) {cleaned_reasoning}"
            # )
        return {self.reasoning_column: cleaned_reasoning}


class MedMCQAAdapter(DatasetAdapter):
    def __call__(self, dataset: DatasetDict, **kwargs) -> (DatasetDict, Dataset):
        return (
            DatasetDict({split: self.format(dset, **kwargs) for split, dset in dataset.items()}),
            None,
        )

    def format(self, dataset: Dataset, **kwargs) -> Dataset:
        # nest the answer options
        dataset = dataset.map(
            NestColumns(["opa", "opb", "opc", "opd"], "answer.text"),
            desc="Cleanup answer options",
            **kwargs
        )
        # cleanup reasoning
        dataset = dataset.map(
            MedMCQACleanupReasoning(reasoning_column="exp"), desc="Cleaning up reasoning", **kwargs
        )

        dataset = dataset.rename_columns(
            {
                "question": "question.text",
                "cop": "answer.target",
                "exp": "question.reasoning",
                "id": "question.uid",
            }
        )

        # cast answer.target
        features = dataset.features.copy()
        features["answer.target"] = datasets.Value("int32")
        dataset = dataset.cast(features=features)

        # drop columns
        dataset = dataset.remove_columns(["opa", "opb", "opc", "opd"])
        return dataset
