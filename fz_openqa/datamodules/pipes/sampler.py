from __future__ import annotations

import math
import warnings
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import loguru
import numpy as np
import rich
import torch
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
        proposal_score_key: str = "proposal_score",
        field="document",
        replace: bool = False,
        largest: bool | Dict[Split, bool] = False,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.total = total
        self.match_score_key = f"{field}.{match_score_key}"
        self.proposal_score_key = f"{field}.{proposal_score_key}"
        self.field = field
        self.replace = replace
        self.largest = largest
        self.temperature = temperature

        super(Sampler, self).__init__(**kwargs)

    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Split = None, **kwargs
    ) -> Batch:
        largest, total, temperature = self._get_args(split)

        logits = batch[self.proposal_score_key]

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

        logits = batch[self.proposal_score_key]

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

        logits = batch[self.proposal_score_key]

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


class PrioritySampler(Sampler):
    """Sample using `priority sampling`: https://arxiv.org/abs/cs/0509026"""

    def __init__(self, *args, from_uniform: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.from_uniform = from_uniform
        loguru.logger.info(
            f"PrioritySampler: from_uniform is set to {self.from_uniform}. "
            f"This means the sampler will consider the base distribution to be uniform"
            f"when computing the importance weights: s(z) = p(z) / tau(z) * u(z) / p(z)."
        )

    @torch.no_grad()
    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Split = None, **kwargs
    ) -> Batch:
        largest, total, temperature = self._get_args(split)

        logits = batch[self.proposal_score_key]
        logits = logits / temperature
        # logits = logits - logits.max(dim=-1, keepdim=True).values
        # batch[self.proposal_score_key] = logits

        # sample
        z, log_w = self.sample(logits, total, largest=largest, from_uniform=self.from_uniform)
        # proposal_log_Z = logits.logsumexp(axis=-1, keepdim=True).expand_as(log_pz)

        # re-index and return
        output = {k: reindex(v, z) for k, v in batch.items()}
        output[f"{self.field}.proposal_log_weight"] = log_w

        return output

    @staticmethod
    def sample(
        logits: torch.Tensor,
        k: int = None,
        largest: bool = False,
        mode: str = "uniform",
        from_uniform: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample `log p(z)` using priority sampling with subset of size `m`

        Args:
            logits (torch.Tensor): un-normalized logits of the distribution
            k (int): size of the subset to sample
            mode (str): sampling mode, one of `uniform`, `exponential`

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: sampled index and log probs
        """

        assert mode in {"uniform", "exponential"}
        log_pz = logits.log_softmax(dim=-1)
        N = log_pz.shape[-1]
        if mode == "uniform":
            if largest:
                u = 0.5 * torch.ones_like(log_pz)
            else:
                u = torch.rand_like(log_pz).clamp(min=1e-12)
        elif mode == "exponential":
            if largest:
                u = torch.ones_like(log_pz)
            else:
                u = torch.empty_like(log_pz)
                u.exponential_()
        else:
            raise ValueError(f"Unknown mode {mode}")

        log_u = u.log()
        keys = log_pz - log_u
        z = keys.argsort(dim=-1, descending=True)[..., : k + 1]
        if k < logits.shape[-1]:
            z_tau = z[..., -1:]
            log_tau = keys.gather(dim=-1, index=z_tau)[..., :1]
        else:
            log_tau = -float("inf") + torch.zeros_like(log_pz)
        z = z[..., :k]
        log_pz = log_pz.gather(dim=-1, index=z)

        if mode == "uniform":
            log_qz = torch.where(log_pz - log_tau < 0, log_pz - log_tau, torch.zeros_like(log_pz))
        elif mode == "exponential":
            log_qz = log_pz - log_tau
            log_qz = (log_qz).exp().mul(-1).exp().mul(-1).log1p()
        else:
            raise ValueError(f"Unknown mode {mode}")

        # compute the log weights, and handle replacing the base distribution with u(z)
        if from_uniform:
            warnings.warn(
                "Priority sampling: consider the base distribution to be uniform"
                "when computing the importance weights: s(z) = p(z) / tau(z) * u(z) / p(z)."
            )
            log_pz = torch.zeros_like(log_pz) - math.log(N)

        # finally, compute the log weights
        log_weight = log_pz - log_qz

        return z, log_weight
