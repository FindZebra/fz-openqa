from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def get_unique_pids(pids: Tensor, fill_value=-1) -> Tensor:
    """
    Get the unique pids across dimension 1 and pad to the max length.
    `torch.unique` sorts the pids in ascending order, so we reverse with -1
    to get the descending order."""
    upids = [torch.unique(r) for r in torch.unbind(-pids)]
    max_length = max(len(r) for r in upids)

    def _pad(r):
        return F.pad(r, (0, max_length - len(r)), value=-fill_value)

    return -torch.stack([_pad(p) for p in upids])
