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
    """Report statistics on the retriever distribution."""

    requires_columns: List[str] = ["document.retrieval_score"]
    output_file_name = "retrieval_distribution.json"
    n_samples: int = 5000
    percentiles: List[int] = [95]
    _allow_wandb: bool = True

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

        # compute the probabilities
        probs = Sampler.compute_probs(scores)

        # compute the entropy
        entropy = -np.sum(probs * np.log(probs), axis=-1)

        # compute probability mass function
        if not all(np.all(s[:-1] >= s[1:]) for s in probs):
            probs = np.concatenate([s[np.argsort(-s)][None] for s in probs], axis=0)
        pmf = np.cumsum(probs, axis=-1)
        assert (
            np.abs(pmf[..., -1] - 1) < 1e-6
        ).all(), f"probabilities don't sum to one: {pmf[..., -1]}"

        # prepare the output and the percentiles for the rank of `p(d|q) == x`
        output = {"entropy": f"{entropy.mean():.3f}", "n_samples": f"{len(scores)}"}
        for p in self.percentiles:
            p = p / 100.0
            arg_pmf = np.argmin(np.abs(pmf - p), axis=-1)
            arg_pmf = arg_pmf.astype(np.float32)
            output[f"rank_pmf={p:.2f}"] = {
                "p50": f"{np.percentile(arg_pmf, 50):.1f}",
                "p95": f"{np.percentile(arg_pmf, 95):.1f}",
                "p99": f"{np.percentile(arg_pmf, 95):.1f}",
            }

        return output
