from numbers import Number
from typing import Dict
from typing import List
from typing import Union

import rich
from torch import Tensor

Batch = Dict[str, Union[Number, Tensor, List[str]]]


def pprint_batch(batch):
    u = ""
    for k, v in batch.items():
        if isinstance(v, Tensor):
            u += f"   - {k}: {v.shape} <{v.dtype}> ({v.device})\n"
        elif isinstance(v, list) and isinstance(v[0], str):
            lens = [len(vv) for vv in v]
            u += f"   - {k}: {min(lens)} to {max(lens)} characters <List(text)>\n"
        else:
            u += f"   - {k}: {v} {type(v)}\n"

    rich.print(u)
