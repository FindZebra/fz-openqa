from typing import Dict

import rich
from torch import Tensor

Batch = Dict[str, Tensor]


def pprint_batch(batch):
    for k, v in batch.items():
        rich.print(f"   - {k}: {v.shape} <{v.dtype}>")
