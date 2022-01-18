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
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.total = total
        self.match_score_key = f"{field}.{match_score_key}"
        self.retrieval_score_key = f"{field}.{retrieval_score_key}"
        self.field = field
        self.replace = replace
        self.largest = largest
        self.temperature = temperature

        super(Sampler, self).__init__(**kwargs)

    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Split = None, **kwargs
    ) -> Batch:
        largest, total, temperature = self._get_args(split)

        logits = batch[self.retrieval_score_key]

        # sample
        index = np.arange(len(logits))
        sampled_index = self._sample(
            index, logits=logits, total=total, largest=largest, temperature=temperature
        )

        # re-index and return
        return {k: reindex(v, sampled_index) for k, v in batch.items()}

    def _sample(
        self,
        index: np.ndarray | List,
        *,
        logits: np.ndarray,
        total: int,
        largest: bool,
        temperature: float,
    ) -> np.ndarray:

        assert temperature > 0, "temperature must be positive"

        if largest:
            sorted_idx = np.argsort(-logits)
            sorted_idx = sorted_idx[:total]
            sampled_index = [index[i] for i in sorted_idx]
        else:
            probs = self.compute_probs(logits, temperature=temperature)
            sampled_index = np.random.choice(index, size=total, p=probs, replace=self.replace)
        return sampled_index

    @staticmethod
    def compute_probs(
        logits: Tensor | np.ndarray,
        temperature: float = 1.0,
        min_prob_value: Optional[float] = 1e-40,
    ) -> np.ndarray:
        if isinstance(logits, Tensor):
            logits = logits.detach().cpu().numpy()

        logits = logits - np.max(logits, axis=-1, keepdims=True)
        logits = np.nan_to_num(logits, nan=-1e6, posinf=1e3, neginf=-1e6)
        probs = softmax(logits / temperature, axis=-1)

        if min_prob_value is not None:
            probs = np.maximum(probs, min_prob_value)
            probs = probs / np.sum(probs, axis=-1, keepdims=True)

        return probs

    def _get_args(self, split):
        total = self.total
        if isinstance(total, (dict, DictConfig)):
            total = total[str(split)]
        largest = self.largest
        if isinstance(largest, (dict, DictConfig)):
            largest = largest[str(split)]
        temperature = self.temperature
        if isinstance(temperature, (dict, DictConfig)):
            temperature = temperature[str(split)]

        return largest, total, temperature


class FirstN(Sampler):
    def __init__(self, *, total: int, **kwargs):
        super().__init__(total=total, **kwargs)

    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Split = None, **kwargs
    ) -> Batch:
        largest, total, temperature = self._get_args(split)
        index = np.arange(total)
        # re-index and return
        return {k: reindex(v, index) for k, v in batch.items()}


class SamplerSupervised(Sampler):
    """Sample"""

    def __init__(self, *args, n_positives: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_positives = n_positives

    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Split = None, **kwargs
    ) -> Batch:
        largest, total, temperature = self._get_args(split)

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
            n = min(self.n_positives, total, len(pos_indexes))
            sampled_pos_indexes = self._sample(
                pos_indexes, logits=pos_logits, total=n, largest=largest, temperature=temperature
            )
        else:
            sampled_pos_indexes = []

        # sample the negative indexes
        n = max(0, total - len(sampled_pos_indexes))
        if len(neg_indexes) > 0 and n > 0:
            neg_logits = logits[neg_indexes]
            n = min(len(neg_indexes), n)
            sampled_neg_indexes = self._sample(
                neg_indexes, logits=neg_logits, total=n, largest=largest, temperature=temperature
            )
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
            sampled_other_indexes = self._sample(
                other_indexes,
                logits=other_logits,
                total=n,
                largest=largest,
                temperature=temperature,
            )

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
        largest, total, temperature = self._get_args(split)

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
            n = min(self.n_boosted, total, len(pos_indexes))
            sampled_pos_indexes = self._sample(
                pos_indexes, logits=pos_logits, total=n, largest=largest, temperature=temperature
            )
        else:
            sampled_pos_indexes = []

        # sample the remaining indexes
        n = max(0, total - len(sampled_pos_indexes))
        other_indexes = [i for i in range(len(logits)) if i not in sampled_pos_indexes]
        if len(other_indexes) > 0 and n > 0:
            other_logits = logits[other_indexes]
            try:
                sampled_other_indexes = self._sample(
                    other_indexes,
                    logits=other_logits,
                    total=n,
                    largest=largest,
                    temperature=temperature,
                )
            except Exception as e:
                rich.print(
                    f"indexes: {len(logits)}, n: {n}, "
                    f"other_indexes: {other_indexes}, "
                    f"sampled_pos_indexes: {sampled_pos_indexes}, "
                    f"other_logits: {other_logits}"
                )
                raise e
        else:
            sampled_other_indexes = []

        # re-index and return
        sampled_indexes = np.concatenate([sampled_pos_indexes, sampled_other_indexes], axis=0)
        sampled_indexes = sampled_indexes.astype(np.long)
        return {k: reindex(v, sampled_indexes) for k, v in batch.items()}
