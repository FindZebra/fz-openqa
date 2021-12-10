from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List

import numpy as np
import rich
from datasets import Dataset
from sklearn.metrics import classification_report
from torch import Tensor

from .base import Analytic
from fz_openqa.utils.pretty import get_separator


def safe_cast_to_list(x: Any) -> Any:
    if isinstance(x, list):
        return [safe_cast_to_list(y) for y in x]
    elif isinstance(x, Tensor):
        if x.dim() == 0:
            return x.item()
        else:
            return [safe_cast_to_list(y) for y in x]


class RetrieverAccuracy(Analytic):
    """Measure the accuracy of the retriever. Only works for concatenated [q; a] datasets."""

    requires_columns: List[str] = ["document.retrieval_score", "answer.target"]
    output_file_name = "retrieval_accuracy.json"

    @staticmethod
    def process_dataset_split(dset: Dataset) -> Dict:
        """
        Report on a specific split of the dataset.
        """
        scores = dset["document.retrieval_score"]
        scores = safe_cast_to_list(scores)
        scores = np.array(scores)
        msg = (
            "Expected scores to be a 3D tensor of shape (batch_size, n_options, n_document). "
            "See ConcatMedQaBuilder."
        )
        assert len(scores.shape) == 3, msg
        targets = np.array(dset["answer.target"])

        # compute predictions based on the retrieval scores
        preds = scores.max(axis=2).argmax(1)

        # report
        labels = list({y for y in targets})
        report_str = classification_report(targets, preds, labels=labels, output_dict=False)
        report_dict = classification_report(targets, preds, labels=labels, output_dict=True)
        return {"dict": report_dict, "str": report_str}

    def _process_results(self, results: Dict[str, Any]):
        """Process the results of the analytic."""
        # log results
        dict_results = {k: r["dict"] for k, r in results.items()}
        self.save_as_json(dict_results)

        if self.verbose:
            print(get_separator())
            rich.print(f"=== {type(self).__name__}  ===")
            for split, r in results.items():
                print(get_separator("."))
                rich.print(f"{split}:")
                rich.print(r["str"])
            print(get_separator())