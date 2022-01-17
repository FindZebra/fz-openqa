from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List

import numpy as np
import rich
from datasets import Dataset
from scipy.special import softmax
from torch import Tensor

from ..pipes import Sampler
from .base import Analytic


def safe_cast_to_list(x: Any) -> Any:
    if isinstance(x, list):
        return [safe_cast_to_list(y) for y in x]
    elif isinstance(x, Tensor):
        if x.dim() == 0:
            return x.item()
        else:
            return [safe_cast_to_list(y) for y in x]


class RetrieverDistribution(Analytic):
    """Report the di"""

    requires_columns: List[str] = ["document.retrieval_score"]
    output_file_name = "retrieval_distribution.json"
    n_samples = 5000
    _allow_wandb: True

    def process_dataset_split(self, dset: Dataset) -> Dict | List:
        """
        Report on a specific split of the dataset.
        """
        scores = dset["document.retrieval_score"]
        scores = safe_cast_to_list(scores)
        scores = np.array(scores)
        scores = scores.reshape(-1, scores.shape[-1])
        if len(scores) > self.n_samples:
            scores = scores[np.random.choice(len(scores), self.n_samples, replace=False)]
        probs = Sampler.compute_probs(scores)

        # compute probability mass function
        if not all(np.all(s[:-1] >= s[1:]) for s in probs):
            probs = np.concatenate([s[np.argsort(-s)][None] for s in probs], axis=0)
        pmf = np.cumsum(probs, axis=-1)

        percentiles = [0.5, 0.9, 0.99]
        output = {}
        for p in percentiles:
            arg_pmf = np.argmin(np.abs(pmf - p), axis=-1)
            arg_pmf = arg_pmf.astype(np.float32)
            output[f"rank_pmf={p:.2f}"] = {
                "n": f"{len(arg_pmf)}",
                "mean": f"{np.mean(arg_pmf):.1f}",
                "std": f"{np.std(arg_pmf):.1f}",
                "p10": f"{np.percentile(arg_pmf, 10):.1f}",
                "p25": f"{np.percentile(arg_pmf, 25):.1f}",
                "p50": f"{np.percentile(arg_pmf, 50):.1f}",
                "p75": f"{np.percentile(arg_pmf, 75):.1f}",
                "p90": f"{np.percentile(arg_pmf, 90):.1f}",
            }

        return output
