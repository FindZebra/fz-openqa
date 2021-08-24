from numbers import Number
from typing import Dict
from typing import List
from typing import Union

import rich
import torch
from torch import Tensor

Batch = Dict[str, Union[Number, Tensor, List[str]]]


def infer_device_from_batch(batch: Batch):
    values = [v for v in batch.values() if isinstance(v, torch.Tensor)]
    if len(values):
        return values[0].device
    else:
        return None


def pprint_batch(batch: Batch):
    u = ""
    for k, v in batch.items():
        if isinstance(v, Tensor):
            u += f"   - {k}: {v.shape} <{v.dtype}> ({v.device})\n"
        elif isinstance(v, list):
            if isinstance(v[0], str):
                lens = [len(vv) for vv in v]
                u += f"   - {k}: {min(lens)} to {max(lens)} characters <list<text>>\n"
            elif isinstance(v[0], list):
                dtype = type(v[0][0]).__name__
                lengths = list(set([len(vv) for vv in v]))
                if len(lengths) == 1:
                    w = f"shape = [{len(v)}, {lengths[0]}]"
                else:
                    w = f"{len(v)} items, each with {min(lengths)} to {max(lengths)} elements"
                u += f"   - {k}: {w} <list<list<{dtype}>>>\n"
        else:
            u += f"   - {k}: {v} {type(v)}\n"

    rich.print(u)
