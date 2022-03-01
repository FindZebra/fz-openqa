import functools
import io
from typing import Any

import torch
import xxhash
from datasets.fingerprint import Hasher
from torch import nn


@functools.singledispatch
def get_fingerprint(obj: Any) -> str:
    """Compute an object fingerprint"""
    hash = Hasher()
    if isinstance(obj, torch.Tensor):
        obj = serialize_tensor(obj)
    hash.update(obj)
    return hash.hexdigest()


def serialize_tensor(x: torch.Tensor):
    x = x.cpu()
    buff = io.BytesIO()
    torch.save(x, buff)
    buff.seek(0)
    return buff.read()


@get_fingerprint.register(nn.Module)
def get_module_weights_fingerprint(obj: nn.Module) -> str:
    hasher = xxhash.xxh64()
    state = obj.state_dict()
    for (k, v) in sorted(state.items(), key=lambda x: x[0]):
        # it did not work without hashing the tensor
        hasher.update(k)
        u = serialize_tensor(v)
        hasher.update(u)

    return hasher.hexdigest()
