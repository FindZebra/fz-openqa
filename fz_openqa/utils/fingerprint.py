import io
from typing import Any

import torch
import xxhash
from datasets.fingerprint import Hasher
from torch import nn


def get_fingerprint(obj: Any) -> str:
    """Compute an object fingerprint"""
    if isinstance(obj, nn.Module):
        hasher = xxhash.xxh64()
        state = obj.state_dict()
        for (k, v) in sorted(state.items(), key=lambda x: x[0]):
            # it did not work without hashing the tensor
            hasher.update(k)
            buff = io.BytesIO()
            torch.save(v, buff)
            buff.seek(0)
            hasher.update(buff.read())

        return hasher.hexdigest()
    else:
        hash = Hasher()
        hash.update(obj)
        return hash.hexdigest()
