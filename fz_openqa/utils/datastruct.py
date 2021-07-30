from numbers import Number
from typing import Dict
from typing import Union

import rich
from torch import Tensor

Batch = Dict[str, Union[Number, Tensor]]


def pprint_batch(batch):
    u = ""
    for k, v in batch.items():
        if isinstance(v, Tensor):
            u += f"   - {k}: {v.shape} <{v.dtype}> ({v.device})\n"
        else:
            u += f"   - {k}: {v} {type(v)}\n"

    rich.print(u)
