from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
import torch
from torch import Tensor

from fz_openqa.datamodules.pipes.base import ApplyToAll
from fz_openqa.datamodules.pipes.base import Pipe
from fz_openqa.utils.datastruct import Batch


def flatten_nested(values: List[List[Any]]) -> List[Any]:
    return [sub_x for x in values for sub_x in x]


def nested_list(values: List[Any], *, stride: int) -> List[List[Any]]:
    output = []
    for i in range(0, len(values), stride):
        output += [[values[j] for j in range(i, i + stride)]]
    return output


def reconcat(values: List[Any], original_type: Type):
    if original_type == Tensor:
        values = torch.cat([t[None] for t in values], dim=0)
    elif original_type == np.ndarray:
        values = np.concatenate([t[None] for t in values], dim=0)
    elif original_type == list:
        pass
    else:
        raise ValueError(
            f"Cannot reconstruct values of original type={original_type}"
        )
    return values


class Flatten(ApplyToAll):
    def __init__(self):
        super().__init__(flatten_nested, element_wise=False)


class Nested(Pipe):
    """Process nested examples using a pipe."""

    def __init__(self, pipe: Pipe, filter: Optional[Callable] = None):
        self.pipe = pipe
        self.filter = filter

    def filter_key(self, k: str) -> bool:
        return self.filter is None or self.filter(k)

    def __call__(self, batch: Batch, **kwargs) -> Batch:

        exs = []
        for i in range(self.batch_size(batch)):
            eg = self.eg(batch, i, filter_op=self.filter)
            eg = self.pipe(eg, **kwargs)
            exs += [eg]

        types = {k: type(v) for k, v in batch.items() if self.filter_key(k)}
        for key in filter(self.filter_key, batch.keys()):
            values = [eg[key] for eg in exs]
            values = reconcat(values, types[key])

            batch[key] = values

        return batch


class Nest(ApplyToAll):
    """Nest a flattened batch:
    {key: [values]} -> {key: [[group] for group in groups]}"""

    def __init__(self, stride: int):
        super(Nest, self).__init__(element_wise=False, op=self.flatten)
        self.stride = stride

    def flatten(self, x: Union[Tensor, List[Any]]):
        if isinstance(x, Tensor):
            return x.view(-1, self.stride, *x.shape[1:])
        else:
            return [
                x[i : i + self.stride] for i in range(0, len(x), self.stride)
            ]
