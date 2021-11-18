from copy import copy
from functools import partial
from typing import List
from typing import Optional
from typing import T
from typing import Union

import numpy as np
from torch import Tensor

from ...utils.pretty import repr_batch
from ...utils.shape import infer_batch_shape
from .base import Pipe
from .utils.nesting import expand_to_shape
from .utils.nesting import flatten_nested
from .utils.nesting import nested_list
from .utils.nesting import reconcat
from fz_openqa.datamodules.pipes.basic import ApplyToAll
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import infer_batch_size


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
    """Flattens the first `level+1` batch dimensions and
    applies the pipe to the flattened batch.

    Warning: Do not use this pipe if the inner pipe drops nested values
    or modifies the order of the batch elements!

    Notes
    -------
    This pipe is equivalent to:

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
    """

    flatten: Optional[Flatten] = None
    nest: Optional[Nest] = None

    def __init__(
        self,
        pipe: Pipe,
        level: int = 1,
        **kwargs,
    ):
        super(ApplyAsFlatten, self).__init__(**kwargs)
        self.pipe = pipe
        self.level = level
        if level > 0:
            self.flatten = Flatten(level=level)
            self.nest = Nest(shape=None)

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        if self.level == 0:
            return self.pipe(batch, **kwargs)

        # infer the original shape of the batch
        input_shape = infer_batch_shape(batch)[: self.flatten.level + 1]
        batch = self.flatten(batch)
        # apply the batch to the flattened batch
        batch = self.pipe(batch, **kwargs)
        # reshape back to the input_shape
        output = self.nest(batch, shape=input_shape)

        # check output and return
        new_shape = infer_batch_shape(output)[: self.flatten.level + 1]
        explain = "Applying a pipe that changes the batch size might have caused this issue."
        if new_shape != input_shape:
            raise ValueError(
                f"{new_shape} != {input_shape}. {explain}\n"
                f"{repr_batch(batch, header='ApplyAsFlatten output batch')}"
            )
        return output


class NestedLevel1(Pipe):
    """
    Apply a pipe to each nested value, handling each nested field as a separate batch.
    This can be use to modify the nested field inplace  (i.e. sorting, deleting).
    However the all pipe output must have the same batch size.
    """

    def __init__(self, pipe: Pipe, **kwargs):
        super().__init__(**kwargs)
        self.pipe = pipe

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        # get initial parameters
        keys = list(batch.keys())
        batch_size = infer_batch_size(batch)
        types = {k: type(v) for k, v in batch.items()}

        # process each Eg separately
        egs = (self.get_eg(batch, idx=i) for i in range(batch_size))
        egs = [self.pipe(eg, **kwargs) for eg in egs]

        # check shape consistency before re-concatenating
        batch_sizes = [infer_batch_size(eg) for eg in egs]
        bs = batch_sizes[0]
        if not all(bs == b for b in batch_sizes):
            raise ValueError(
                f"Batch sizes are inconsistent. "
                f"Make sure the pipe {type(self.pipe)} returns "
                f"the same batch size for all nested examples."
            )

        # concatenate and return
        return {key: reconcat([eg[key] for eg in egs], types[key]) for key in keys}


class Nested(ApplyAsFlatten):
    """
    Apply a pipe to each nested value up to dimension `level`.
    This can be use to modify the nested field inplace  (i.e. sorting, deleting).
    However the all pipe output must have the same batch size.
    """

    def __init__(self, pipe: Pipe, level=1, **kwargs):
        """
        Parameters
        ----------
        pipe
            The pipe to apply to each nested batch.
        level
            The level of nesting to apply the pipe to.
        kwargs
            Additional keyword arguments passed to `ApplyAsFlatten`.
        """
        pipe = NestedLevel1(pipe)
        super().__init__(pipe=pipe, level=level - 1, **kwargs)


class Expand(Pipe):
    """
    Expand the batch to match the new shape. New dimensions are repeated.
    """

    def __init__(self, shape: List[int], **kwargs):
        """
        Parameters
        ----------
        shape
            The shape of the batch after expansion.
        kwargs
            Additional keyword arguments passed to `Pipe`.
        """
        super().__init__(**kwargs)
        self.shape = shape

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        target_shape = copy(self.shape)
        shape, shapes = infer_batch_shape(batch, return_all_shapes=True)

        # replace negative target_shape values with the original shape
        for i, s in enumerate(shape):
            if target_shape[i] < 0:
                target_shape[i] = s

        # check shape consistency
        if not all(list(shape) == list(s) for s in shapes.values()):
            raise ValueError(f"All fields must have the same shape. Found: {shapes}")
        if not list(target_shape[: len(shape)]) == list(shape):
            raise ValueError(
                f"First dimensions must match. "
                f"Cannot expand batch of shape {shape} to shape {self.shape}"
            )
        if shape == target_shape:
            return batch
        else:
            return {k: expand_to_shape(v, target_shape) for k, v in batch.items()}
