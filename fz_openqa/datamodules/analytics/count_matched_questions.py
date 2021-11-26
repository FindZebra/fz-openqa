from __future__ import annotations

from typing import Dict
from typing import List

from datasets import Dataset
from datasets import DatasetDict

from ..utils.datastruct import OpenQaDataset
from ..utils.typing import HfDataset
from .base import Analytic
from fz_openqa.datamodules.utils.dataset import get_column_names


class CountMatchedQuestions(Analytic):
    """Count the number of questions matched with positive documents"""

    requires_columns: List[str] = ["document.match_score"]
    output_file_name: str = "count_matched_questions.json"

    def process_dataset_split(self, dset: Dataset) -> Dict:
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
