from functools import partial
from typing import Callable
from typing import List
from typing import Optional
from typing import T
from typing import Union

import numpy as np
from torch import Tensor

from .base import Pipe
from .utils.nesting import flatten_nested
from .utils.nesting import nested_list
from .utils.nesting import reconcat
from fz_openqa.datamodules.pipes.basic import ApplyToAll
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import always_true


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
        if level < 1:
            raise ValueError("level must be >= 1")
        self.level = level
        fn = partial(flatten_nested, level=level)
        super().__init__(fn, element_wise=False, **kwargs)


class Nest(ApplyToAll):
    """
    Nest a flat batch. This is equivalent to calling np.reshape to all values,
    except that this method can handle np.ndarray, Tensors and lists.
    If the target shape is unknown at initialization time, the `shape` attributed
    can be passed as a keyword argument to the __call__ method.
    """

    def __init__(self, shape: Optional[List[int]], **kwargs):
        """
        Parameters
        ----------
        shape
            Target shape of the nested batch (e.g. [-1, 2, 4]), potentially unknown at init.
        kwargs
            Forwarded to ApplyToAll
        """
        nest_fn = partial(self.nest, _shape=shape)
        super(Nest, self).__init__(element_wise=False, op=nest_fn, allow_kwargs=True, **kwargs)

    @staticmethod
    def nest(
        x: T, *, _shape: Optional[List[int]], shape: Optional[List[int]] = None, **kwargs
    ) -> T:
        """
        Nest the input x according to shape or _shape.
        This allows specifying a shape that is not known at init.

        Parameters
        ----------
        x
            Input to nest.
        shape
            Primary and optional target shape of the nested batch
        _shape
            Secondary and optional target shape of the nested batch

        Returns
        -------
        Union[List, Tensor, np.ndarray]
            Nested input.

        """
        shape = shape or _shape
        if shape is None:
            raise ValueError("Either shape or _shape must be provided")

        if isinstance(x, Tensor):
            return x.view(*shape, *x.shape[1:])
        elif isinstance(x, np.ndarray):
            return x.reshape((*shape, *x.shape[1:]))
        elif isinstance(x, list):
            return nested_list(x, shape=shape)
        else:
            raise TypeError(f"Unsupported type: {type(x)}")


class ApplyAsFlatten(Pipe):
    """Flatten nested field an apply a pipe to the flattened batch.
    This is equivalent to:

    ```python
    # example data
    h = (20, 10) # some vector dimension
    nested_shape = (10, 8, 8) # some nested batch dimension
    batch = np.random.randn(size=([nested_shape, *h)]

    # ApplyAsFlatten(pipe)
    batch = batch.reshape(-1, *h)
    batch = pipe(batch)
    batch = batch.reshape(*nested_shape, *h)
    ```

    Warning: Do not use this pipe if the inner pipe modify the order of the batch elements!
    """

    def __init__(
        self,
        pipe: Pipe,
        flatten_level: int = 1,
        **kwargs,
    ):
        super(ApplyAsFlatten, self).__init__(**kwargs)
        self.pipe = pipe
        self.flatten = Flatten(level=flatten_level)
        self.nest = Nest(shape=None)

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        raise NotImplementedError


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
