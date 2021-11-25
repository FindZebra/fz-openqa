import json
from typing import Dict
from typing import Union

import rich
from datasets import Dataset
from datasets import DatasetDict

from ...utils.pretty import get_separator
from ..utils.datastruct import OpenQaDataset
from ..utils.typing import HfDataset
from .base import Analytic
from fz_openqa.datamodules.utils.dataset import get_column_names


class CountMatchedQuestions(Analytic):
    """Count the number of questions matched with positive documents"""

    @staticmethod
    def report_split(dset: Dataset) -> Dict:
        """
        Report on a specific split of the dataset.
        """
        n_scores = dset["document.retrieval_score"]
        n_pos = dset["document.match_score"]
        scores = [y[0].item() for x, y in zip(n_pos, n_scores) if sum(x) > 0]
        matches = [sum(y) for y in n_pos]
        n = len(matches)
        count = len([y for y in matches if y > 0])
        return {
            "total": n,
            "count": count,
            "ratio": count / n,
            "avg_score": sum(scores) / len(scores),
        }

    def __call__(self, dataset: Union[HfDataset, OpenQaDataset], **kwargs):
        """Count the number of questions matched with positive documents."""
        assert "document.match_score" in get_column_names(dataset)
        if isinstance(dataset, DatasetDict):
            results = {split: self.report_split(dset) for split, dset in dataset.items()}
        elif isinstance(dataset, Dataset):
            results = {"all": self.report_split(dataset)}
        else:
            raise TypeError(f"Unsupported type {type(dataset)}")

        # log results
        self.save_as_json(results, "count_matched_questions.json")

        if self.verbose:
            print(get_separator())
            rich.print(f"=== {type(self).__name__} ===")
            print(get_separator("."))
            rich.print(json.dumps(results, indent=2))
            print(get_separator())
