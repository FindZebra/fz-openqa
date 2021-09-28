from numbers import Number
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import torch
from torch import Tensor

Batch = Dict[str, Union[Number, Tensor, List[str]]]


def infer_device_from_batch(batch: Batch):
    values = [v for v in batch.values() if isinstance(v, torch.Tensor)]
    if len(values):
        return values[0].device
    else:
        return None


def add_prefix(d: Dict[str, Any], prefix: str):
    return {f"{prefix}{k}": v for k, v in d.items()}


def filter_prefix(d: Dict[str, Any], prefix: str):
    return {k.replace(prefix, ""): v for k, v in d.items() if prefix in k}


def contains_prefix(key, output):
    return any(key in k for k in output.keys())
