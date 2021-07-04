from typing import Dict

import rich
from torch import Tensor

Batch = Dict[str, Tensor]


def pprint_batch(batch):
    u = ""
    for k, v in batch.items():
        u += f"   - {k}: {v.shape} <{v.dtype}> ({v.device})\n"

    rich.print(u)
