from __future__ import annotations

import warnings
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import rich
from datasets import Dataset
from datasets import Split
from sklearn.metrics import classification_report
from torch import Tensor
from tqdm import tqdm

from ..utils.dataset import keep_only_columns
from .base import Analytic
from .retriever_distribution import safe_concatenate
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
    batch_size: int = 1000
    _allow_wandb: True

    def __init__(self, *args, method: str = "sum", **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method

    def process_dataset_split(
        self, dset: Dataset, *, split: Optional[str | Split] = None
    ) -> Dict | List:
        """
        Report on a specific split of the dataset.
        """
        all_preds = None
        all_targets = None
        dset = keep_only_columns(dset, columns=["document.retrieval_score", "answer.target"])
        for i in tqdm(
            range(0, len(dset), self.batch_size),
            desc=f"{type(self).__name__}, split={split}",
            disable=not self.verbose,
            leave=False,
        ):
            row = dset[i : i + self.batch_size]
            scores = row["document.retrieval_score"]
            scores = safe_concatenate(scores)
            msg = (
                "Expected scores to be a 3D tensor of shape (batch_size, n_options, n_document). "
                "See ConcatMedQaBuilder."
            )
            assert len(scores.shape) == 3, msg

            targets = np.array(row["answer.target"])

            # compute predictions based on the retrieval scores
            logits = self._reduce(scores)
            preds = logits.argmax(1)

            # concatenaate the results
            if all_preds is None:
                all_preds = preds
                all_targets = targets
            else:
                all_preds = np.concatenate([all_preds, preds], axis=-1)
                all_targets = np.concatenate([all_targets, targets], axis=-1)

        # report
        labels = list(sorted({y for y in all_targets}))
        report_str = classification_report(
            all_targets, all_preds, labels=labels, output_dict=False, digits=3
        )
        report_dict = classification_report(
            all_targets, all_preds, labels=labels, output_dict=True, digits=3
        )
        report_dict = {k: v for k, v in report_dict.items() if k in ["macro avg", "accuracy"]}
        rich.print(report_dict)

        return {"dict": report_dict, "str": report_str}

    def _reduce(self, scores):
        if self.method == "sum":
            logits = scores.sum(axis=-1)
        elif self.method == "max":
            logits = scores.max(axis=-1)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return logits

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

        if self.wandb_log:
            try:
                self.log_to_wandb(dict_results)
            except Exception as e:
                warnings.warn(f"Could not log to wandb: {e}")
