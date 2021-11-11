from functools import partial
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from torch import Tensor

from .base import Pipe
from .utils.nesting import flatten_nested
from .utils.nesting import nested_list
from .utils.nesting import reconcat
from fz_openqa.datamodules.pipes.basic import ApplyToAll
from fz_openqa.datamodules.pipes.basic import FilterKeys
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import always_true
from fz_openqa.utils.functional import infer_batch_size
from fz_openqa.utils.functional import infer_stride

STRIDE_SYMBOL = "__stride__"


class Flatten(ApplyToAll):
    """
    Flatten a nested batch up to dimension=`level`.
    For instance a batch of shape (x, 3, 4, ...) with level=2 will be flattened to (x *3 * 4, ...)
    """

    def __init__(self, level: int = 1, **kwargs):
        """
        Parameters
        ----------
        level
            The level to flatten to.
        kwargs
            Keyword arguments passed to `ApplyToAll`.
        """
        fn = partial(flatten_nested, level=level)
        super().__init__(fn, element_wise=False, **kwargs)


class Nest(ApplyToAll):
    """
    Nest a flat batch. This is equivalent to calling np.reshape to all values,
    except that this method can handle np.ndarray, Tensors and lists
    """

    def __init__(self, shape: List[int], **kwargs):
        """
        Parameters
        ----------
        shape
            Target shape of the nested batch (e.g. [-1, 2, 4])
        kwargs
            Forwarded to ApplyToAll
        """
        nest_fn = partial(self.nest, shape=shape)
        super(Nest, self).__init__(element_wise=False, op=nest_fn, allow_kwargs=False, **kwargs)

    @staticmethod
    def nest(x: Union[Tensor, List[Any]], *, shape: List[int]):
        if isinstance(x, Tensor):
            return x.view(*shape, *x.shape[1:])
        elif isinstance(x, np.ndarray):
            return x.reshape((*shape, *x.shape[1:]))
        elif isinstance(x, list):
            return nested_list(x, shape=shape)
        else:
            raise TypeError(f"Unsupported type: {type(x)}")


class Nested(Pipe):
    """
    Apply a pipe to each nested value.
    This can be use to modify the nested field inplace  (i.e. sorting, deleting).
    # todo: stopping here, continue tomorrow
    """

    def __init__(self, pipe: Pipe, filter: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self.pipe = pipe
        self.filter = filter or always_true

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:

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

    Warning: Do not use this pipe if the inner pipe modify the order of the batch elements!

     # todo: stopping here, continue tomorrow
    """

    def __init__(
        self,
        pipe: Pipe,
        input_filter: Optional[Callable] = None,
        update: bool = False,
        **kwargs,
    ):
        super(ApplyAsFlatten, self).__init__(**kwargs)
        self.pipe = pipe
        self.update = update
        self.flatten = Flatten()
        self.nest = Nest(stride=None)
        self.filter = FilterKeys(input_filter)

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
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
