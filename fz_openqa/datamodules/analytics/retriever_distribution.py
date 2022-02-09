from __future__ import annotations

import logging
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from datasets import Split
from tqdm import tqdm

from ..pipes import Sampler
from ..utils.dataset import keep_only_columns
from .base import Analytic


def safe_concatenate(scores):
    if isinstance(scores, list):
        if isinstance(scores[0], (torch.Tensor, np.ndarray)):
            if isinstance(scores[0], torch.Tensor):
                scores = [s.cpu().numpy() for s in scores]
            return np.concatenate([s[None] for s in scores])
        else:
            scores = [safe_concatenate(s) for s in scores]
            return safe_concatenate(scores)
    else:
        return scores


class RetrieverDistribution(Analytic):
    """Report statistics on the retriever distribution."""

    requires_columns: List[str] = ["document.retrieval_score"]
    output_file_name = "retrieval_distribution.json"
    n_samples: int = 1000
    batch_size: int = 1000
    percentiles: List[int] = [10, 50, 90]
    _allow_wandb: bool = True

    def process_dataset_split(
        self, dset: Dataset, *, split: Optional[str | Split] = None
    ) -> Dict | List:
        """
        Report on a specific split of the dataset.
        """
        scores = None
        dset = keep_only_columns(dset, ["document.retrieval_score"])
        n_steps = len(dset) // self.batch_size
        for i in tqdm(
            range(0, len(dset), self.batch_size),
            desc=f"{type(self).__name__}, split={split}",
            disable=not self.verbose,
            leave=False,
        ):
            row = dset[i : i + self.batch_size]
            scores_i = self._get_scores(
                row, key="document.retrieval_score", n_samples=self.n_samples // max(1, n_steps)
            )
            if scores is None:
                scores = scores_i
            else:
                scores = np.concatenate([scores, scores_i])[: self.n_samples]

        # save scores to file
        self._save_array(scores, "scores", split)

        # compute the probabilities
        probs = Sampler.compute_probs(scores)

        # save probs to file
        self._save_array(probs, "probs", split)

        # compute the entropy
        entropy = -np.sum(probs * np.log(probs), axis=-1)

        # compute probability mass function
        if not all(np.all(s[:-1] >= s[1:]) for s in probs):
            probs = np.concatenate([s[np.argsort(-s)][None] for s in probs], axis=0)
        pmf = np.cumsum(probs, axis=-1)
        assert (
            np.abs(pmf[..., -1] - 1) < 1e-3
        ).all(), f"probabilities don't sum to one: {pmf[..., -1]}"

        # prepare the output and the percentiles for the rank of `p(d|q) == x`
        min_max_scores = np.min(scores, axis=-1), np.max(scores, axis=-1)
        output = {
            "entropy": f"{entropy.mean():.3f}",
            "n_samples": f"{len(scores)}",
            "min-max-scores": f"{np.mean(min_max_scores):.3f}",
        }
        for p in self.percentiles:
            p = p / 100.0
            arg_pmf = np.argmin(np.abs(pmf - p), axis=-1)
            arg_pmf = arg_pmf.astype(np.float32)
            output[f"rank_pmf={p:.2f}"] = {
                "p50": f"{np.percentile(arg_pmf, 50):.1f}",
            }

        return output

    def _save_array(self, probs, name, split):
        sub_directory = self.output_dir / self.output_file_name.replace(".json", "")
        path = sub_directory / f"{name}-{split}.npy"
        path = self.indexed_output_file(path)
        np.save(str(path), probs)
        logging.getLogger(__name__).info(f"saved {name} to {path}")

    def _get_scores(self, row: Dict | Dataset, *, key: str, n_samples: int) -> np.ndarray:
        scores = row[key]
        scores = safe_concatenate(scores)
        scores = scores.reshape(-1, scores.shape[-1])
        if len(scores) > n_samples:
            scores = scores[np.random.choice(len(scores), n_samples, replace=False)]
        return scores
