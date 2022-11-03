from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
from torch import Tensor


Array = Union[List[List], np.ndarray, Tensor]


def concat_arrays(*a: Array, dim=0) -> Array:
    arr_type = type(a[0])
    assert all(isinstance(x, arr_type) for x in a)
    if arr_type == list:
        if dim == 0:
            return sum(a, [])
        elif dim == 1:
            new_array = []
            assert all(len(x) == len(a[0]) for x in a)
            for i in range(len(a[0])):
                new_array.append(sum([x[i] for x in a], []))
            return new_array
    elif arr_type == np.ndarray:
        return np.concatenate(a, axis=dim)
    elif arr_type == Tensor:
        return torch.cat(a, dim=dim)
    else:
        raise TypeError(f"Unsupported type: {type(a[0])}")
