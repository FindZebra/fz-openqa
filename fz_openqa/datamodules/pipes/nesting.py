from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
import torch
from torch import Tensor

from fz_openqa.datamodules.pipes.base import always_true
from fz_openqa.datamodules.pipes.base import ApplyToAll
from fz_openqa.datamodules.pipes.base import FilterKeys
from fz_openqa.datamodules.pipes.base import Pipe
from fz_openqa.utils.datastruct import Batch

STRIDE_SYMBOL = "__stride__"


def infer_batch_size(batch: Batch) -> int:
    def _cond(k):
        return not k.startswith("__") and not k.endswith("__")

    return next(iter((len(v) for k, v in batch.items() if _cond(k))))


def infer_stride(batch: Batch) -> int:
    def _cond(k):
        return not k.startswith("__") and not k.endswith("__")

    x = next(iter(v for k, v in batch.items() if _cond(k)))
    stride = len(next(iter(x)))
    assert all(stride == len(y) for y in x)
    return stride


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


class Nest(ApplyToAll):
    """Nest a flattened batch:
    {key: [values]} -> {key: [[group] for group in groups]}"""

    def __init__(self, stride: Optional[int]):
        super(Nest, self).__init__(
            element_wise=False, op=self.nest, allow_kwargs=True
        )
        self.stride = stride

    def nest(
        self,
        x: Union[Tensor, List[Any]],
        stride: Optional[int] = None,
        **kwargs,
    ):
        stride = stride or self.stride
        if isinstance(x, Tensor):
            return x.view(-1, stride, *x.shape[1:])
        else:
            return [x[i : i + stride] for i in range(0, len(x), stride)]


class Nested(Pipe):
    """
    Apply a pipe to each nested value.
    This can be use to modify the nested field inplace  (i.e. sorting, deleting).
    """

    def __init__(
        self, pipe: Pipe, filter: Optional[Callable] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.pipe = pipe
        self.filter = filter or always_true

    def __call__(self, batch: Batch, **kwargs) -> Batch:

        exs = []
        for i in range(self.batch_size(batch)):
            eg = self.get_eg(batch, i, filter_op=self.filter)
            eg = self.pipe(eg, **kwargs)
            exs += [eg]

        types = {k: type(v) for k, v in batch.items() if self.filter(k)}
        for key in filter(self.filter, batch.keys()):
            values = [eg[key] for eg in exs]
            values = reconcat(values, types[key])

            batch[key] = values

        return batch


class ApplyAsFlatten(Pipe):
    """Flatten nested field an apply a pipe to the flatten fields.

    Warning: Do not use this pipe if the inner pipe modify the order of the batch elements!"""

    def __init__(
        self,
        pipe: Pipe,
        filter: Optional[Callable] = None,
        update: bool = False,
        **kwargs,
    ):
        super(ApplyAsFlatten, self).__init__(**kwargs)
        self.pipe = pipe
        self.update = update
        self.flatten = Flatten()
        self.nest = Nest(stride=None)
        self.filter = FilterKeys(filter)

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        output = self._filter_and_apply(batch, **kwargs)
        if self.update:
            batch.update(output)
            return batch
        else:
            return output

    def _filter_and_apply(self, batch: Batch, **kwargs) -> Batch:
        batch = self.filter(batch)
        batch_size = infer_batch_size(batch)
        stride = infer_stride(batch)
        batch = self.flatten(batch)
        batch = self.pipe(batch, **kwargs)
        output = self.nest(batch, stride=stride)

        new_batch_size = infer_batch_size(output)
        new_stride = infer_stride(output)
        assert batch_size == new_batch_size
        assert new_stride == stride

        return output

    def fingerprint(self):
        return {
            "filter": self.filter.fingerprint(),
            "flatten": self.flatten.fingerprint(),
            "nest": self.nest.fingerprint(),
            type(self.pipe).__name__: self.pipe.fingerprint(),
        }
