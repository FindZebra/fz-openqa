from __future__ import annotations

from typing import List

import rich
import torch
from loguru import logger
from torch import Tensor
from torch.nn import functional as F

from fz_openqa.datamodules.index.maxsim.datastruct import MaxSimOutput
from fz_openqa.datamodules.index.maxsim.datastruct import MaxSimStats


class MaxSimReducer(object):
    """Reduce the `MaxSimOutput`s from each `MaxSimWorker` to a single `MaxSimOutput`"""

    def __init__(self, device: None | int | torch.device = None):
        if device is None:
            device = torch.device("cpu")
        if isinstance(device, int):
            device = torch.device("cuda", device)
        self.device = device

    def __call__(self, data: List[MaxSimOutput], k: int) -> MaxSimOutput:
        assert isinstance(data, list)
        assert all(isinstance(d, MaxSimOutput) for d in data)

        # gather data
        scores = [d.scores for d in data]
        pids = [d.pids for d in data]

        # send to device
        devices = list(set(s.device for s in scores))
        if len(devices) > 1:
            if self.device is None:
                device = devices[0]
            else:
                device = self.device
            scores = [s.to(device, non_blocking=True) for s in scores]
            pids = [p.to(device, non_blocking=True) for p in pids]

        # concatenate
        all_scores, all_pids = (torch.cat(x, dim=-1) for x in (scores, pids))

        # debug
        upids = torch.unique(all_pids)
        prop_unique = upids.numel() / all_pids.numel() if all_pids.numel() > 0 else -1
        number_docs_per_row = (all_pids >= 0).float().sum(dim=1)

        # take the top-k results given the MaxSim score
        k_ = min(k, all_scores.shape[-1])
        if k_ < k:
            logger.warning(f"Could't retriever {k} results, found {k_}")

        all_scores = all_scores.to(torch.float32)
        maxsim_idx = torch.topk(all_scores, k=k_, dim=-1, largest=True, sorted=True).indices

        # fetch the corresponding document indices and return
        maxsim_scores = all_scores.gather(index=maxsim_idx, dim=-1)
        maxsim_pids = all_pids.gather(index=maxsim_idx, dim=-1)

        assert maxsim_pids.shape[-1] > 0

        if maxsim_scores.shape[1] < k or maxsim_pids.shape[1] < k:
            maxsim_pids, maxsim_scores = self._pad_outputs(k, maxsim_pids, maxsim_scores)
        output = MaxSimOutput(
            scores=maxsim_scores.to(self.device, non_blocking=True),
            pids=maxsim_pids.to(self.device, non_blocking=True),
            k=k,
            boundaries=None,
            stats=MaxSimStats(prop_unique=prop_unique, number_docs_per_row=number_docs_per_row),
        )
        return output

    def cleanup(self):
        pass

    @staticmethod
    def _pad_outputs(k: int, maxsim_pids: Tensor, maxsim_scores: Tensor):
        # pad maxsim_scores with nans
        maxsim_scores = MaxSimReducer._pad_to_length(maxsim_scores, k, -torch.inf)
        # pad maxsim_pids with zeros
        maxsim_pids = MaxSimReducer._pad_to_length(maxsim_pids, k, -1)
        return maxsim_pids, maxsim_scores

    @staticmethod
    def _pad_to_length(values: Tensor, k: int, fill_value=torch.tensor(float("nan"))):
        if values.shape[1] < k:
            return F.pad(values, (0, k - values.shape[1]), value=fill_value)
        else:
            return values
