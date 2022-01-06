from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import rich
from datasets import Split
from omegaconf import DictConfig
from scipy.special import softmax
from torch import Tensor

from .base import Pipe
from .sorting import reindex
from fz_openqa.utils.datastruct import Batch


class Sampler(Pipe):
    def __init__(
        self,
        *,
        total: int | Dict[Split, int],
        match_score_key: str = "match_score",
        retrieval_score_key: str = "retrieval_score",
        field="document",
        replace: bool = False,
        largest: bool | Dict[Split, bool] = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.total = total
        self.match_score_key = f"{field}.{match_score_key}"
        self.retrieval_score_key = f"{field}.{retrieval_score_key}"
        self.field = field
        self.replace = replace
        self.largest = largest

        super(Sampler, self).__init__(**kwargs)

    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Split = None, **kwargs
    ) -> Batch:
        largest, total = self._get_args(split)

        logits = batch[self.retrieval_score_key]
        probs = self.compute_probs(logits)

        # sample
        index = np.arange(len(probs))
        sampled_index = self._sample(index, probs, total, largest=largest)

        # re-index and return
        return {k: reindex(v, sampled_index) for k, v in batch.items()}

    def _sample(self, index: np.ndarray | List, probs: np.ndarray, total: int, largest=None):
        largest = largest or self.largest
        if largest:
            sorted_idx = np.argsort(-probs)
            sorted_idx = sorted_idx[:total]
            sampled_index = [index[i] for i in sorted_idx]
        else:
            sampled_index = np.random.choice(index, size=total, p=probs, replace=self.replace)
        return sampled_index

    @staticmethod
    def compute_probs(
        logits: Tensor | np.ndarray, min_value: Optional[float] = 1e-40
    ) -> np.ndarray:
        if isinstance(logits, Tensor):
            logits = logits.detach().cpu().numpy()
        logits = np.nan_to_num(logits, nan=-1e3, posinf=1e3, neginf=-1e3)
        probs = softmax(logits)

        if min_value is not None:
            probs = np.maximum(probs, min_value)
            probs = probs / np.sum(probs)
        return probs

    def _get_args(self, split):
        total = self.total
        if isinstance(total, (dict, DictConfig)):
            total = total[str(split)]
        largest = self.largest
        if isinstance(largest, (dict, DictConfig)):
            largest = largest[str(split)]
        return largest, total


class SamplerSupervised(Sampler):
    """Sample"""

    def __init__(self, *args, n_positives: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_positives = n_positives

    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Split = None, **kwargs
    ) -> Batch:
        largest, total = self._get_args(split)

        logits = batch[self.retrieval_score_key]

        if self.match_score_key in batch.keys():
            scores = batch[self.match_score_key]
            # select positives samples
            pos_indexes = [i for i, x in enumerate(scores) if x > 0]
            neg_indexes = [i for i, x in enumerate(scores) if x <= 0]
        else:
            pos_indexes = []
            neg_indexes = []

        # sample positives
        if self.n_positives > 0 and len(pos_indexes) > 0:
            pos_logits = logits[pos_indexes]
            pos_probs = self.compute_probs(pos_logits)
            n = min(self.n_positives, total, len(pos_indexes))
            sampled_pos_indexes = self._sample(pos_indexes, pos_probs, n, largest=largest)
        else:
            sampled_pos_indexes = []

        # sample the negative indexes
        n = max(0, total - len(sampled_pos_indexes))
        if len(neg_indexes) > 0 and n > 0:
            neg_logits = logits[neg_indexes]
            neg_probs = self.compute_probs(neg_logits)
            n = min(len(neg_indexes), n)
            sampled_neg_indexes = self._sample(neg_indexes, neg_probs, n, largest=largest)
        else:
            sampled_neg_indexes = []

        # concatenate positive and negative samples
        sampled_indexes = np.concatenate([sampled_pos_indexes, sampled_neg_indexes], axis=0)
        sampled_indexes = sampled_indexes.astype(np.long)

        # sample remaining indexes if needed
        sampled_other_indexes = []
        if len(sampled_indexes) < total:
            n = max(0, total - len(sampled_indexes))
            other_indexes = [i for i in range(len(logits)) if i not in sampled_indexes]
            assert len(other_indexes) >= n
            other_logits = logits[other_indexes]
            other_probs = self.compute_probs(other_logits)
            sampled_other_indexes = self._sample(other_indexes, other_probs, n, largest=largest)

        # re-index and return
        sampled_indexes = np.concatenate([sampled_indexes, sampled_other_indexes], axis=0)
        sampled_indexes = sampled_indexes.astype(np.long)
        return {k: reindex(v, sampled_indexes) for k, v in batch.items()}


class SamplerBoostPositives(Sampler):
    """Sample first `n_boosted` positive samples from the batch and
    complete with any other sample"""

    def __init__(self, *args, n_boosted: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_boosted = n_boosted

    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Split = None, **kwargs
    ) -> Batch:
        largest, total = self._get_args(split)

        logits = batch[self.retrieval_score_key]

        if self.match_score_key in batch.keys():
            scores = batch[self.match_score_key]
            # select positives samples
            pos_indexes = [i for i, x in enumerate(scores) if x > 0]
        else:
            pos_indexes = []

        # sample positives
        if self.n_boosted > 0 and len(pos_indexes) > 0:
            pos_logits = logits[pos_indexes]
            pos_probs = self.compute_probs(pos_logits)
            n = min(self.n_boosted, total, len(pos_indexes))
            sampled_pos_indexes = self._sample(pos_indexes, pos_probs, n, largest=largest)
        else:
            sampled_pos_indexes = []

        # sample the remaining indexes
        n = max(0, total - len(sampled_pos_indexes))
        other_indexes = [i for i in range(len(logits)) if i not in sampled_pos_indexes]
        if len(other_indexes) > 0 and n > 0:
            other_logits = logits[other_indexes]
            other_probs = self.compute_probs(other_logits)
            try:
                sampled_other_indexes = self._sample(other_indexes, other_probs, n, largest=largest)
            except Exception as e:
                rich.print(
                    f"indexes: {len(logits)}, n: {n}, "
                    f"other_indexes: {other_indexes}, "
                    f"sampled_pos_indexes: {sampled_pos_indexes}, "
                    f"other_probs: {other_probs}, "
                    f"other_logits: {other_logits}"
                )
                raise e
        else:
            sampled_other_indexes = []

        # re-index and return
        sampled_indexes = np.concatenate([sampled_pos_indexes, sampled_other_indexes], axis=0)
        sampled_indexes = sampled_indexes.astype(np.long)
        return {k: reindex(v, sampled_indexes) for k, v in batch.items()}
