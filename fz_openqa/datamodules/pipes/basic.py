from copy import copy
from copy import deepcopy
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import rich

from .base import Pipe
from .control.condition import Condition
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import Eg


class Identity(Pipe):
    """
    A pipe that passes a batch without modifying it.
    """

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        """Returns the batch without modifying it."""
        return batch

    def _call_egs(self, batch: Batch, **kwargs) -> Batch:
        """Returns the batch without modifying it."""
        return batch


class Lambda(Pipe):
    """
    Apply a lambda function to the batch.
    """

    def __init__(
        self,
        op: Callable,
        output_keys: Optional[List[str]] = None,
        allow_kwargs: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.op = op
        self._output_keys = output_keys
        self.allow_kwargs = allow_kwargs

    def _call_batch(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> Batch:
        return self._call_all(batch, **kwargs)

    def _call_egs(self, examples: List[Eg], idx: Optional[List[int]] = None, **kwargs) -> Batch:
        return self._call_all(examples, **kwargs)

    def _call_all(self, batch: Union[List[Eg], Batch], **kwargs) -> Batch:
        """The call of the pipeline process"""
        if not self.allow_kwargs:
            kwargs = {}
        return self.op(batch, **kwargs)

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return self._output_keys or super().output_keys(input_keys)


class GetKey(Pipe):
    """
    Returns a batch containing only the target key.
    """

    def __init__(self, key: str, **kwargs):
        super().__init__(**kwargs)
        self.key = key

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        return {self.key: batch[self.key]}

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return [self.key]


class FilterKeys(Identity):
    """
    Filter the keys in the batch given the `Condition` object.
    """

    _allows_update = False

    def __init__(self, condition: Optional[Condition], **kwargs):
        assert kwargs.get("input_filter", None) is None, "input_filter is not allowed"
        super().__init__(input_filter=condition, **kwargs)


class DropKeys(Pipe):
    """
    Drop the keys in the current batch.
    """

    _allows_update = False
    _allows_input_filter = False

    def __init__(self, keys: List[str], **kwargs):
        super().__init__(**kwargs)
        self.keys = keys

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        for key in self.keys:
            batch.pop(key)
        return batch

    def output_keys(self, input_keys: List[str]) -> List[str]:
        for key in self.keys:
            input_keys.remove(key)
        return input_keys


class AddPrefix(Pipe):
    """
    Append the keys with a prefix.
    """

    _allows_update = False

    def __init__(self, prefix: str, **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        return {f"{self.prefix}{k}": v for k, v in batch.items()}

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return [f"{self.prefix}{k}" for k in input_keys]


class ReplaceInKeys(Pipe):
    """
    Remove a pattern `a` with `b` in all keys
    """

    _allows_update = False

    def __init__(self, a: str, b: str, **kwargs):
        """
        Parameters
        ----------
        a
            pattern to be replaced
        b
            pattern to replace with
        kwargs
            Other Parameters
        """
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        return {k.replace(self.a, self.b): v for k, v in batch.items()}

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return [k.replace(self.a, self.b) for k in input_keys]


class RenameKeys(Pipe):
    """
    Rename a set of keys using a dictionary
    """

    _allows_update = False

    def __init__(self, keys: Dict[str, str], **kwargs):
        """
        Parameters
        ----------
        keys
            A dictionary mapping old keys to new keys.
        kwargs
            Other Parameters
        """
        super().__init__(**kwargs)
        self.keys = keys

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        for old_key, new_key in self.keys.items():
            if old_key in batch:
                value = batch.pop(old_key)
                batch[new_key] = value

        return batch

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return [self.keys.get(k, k) for k in input_keys]


class Apply(Pipe):
    """
    Transform the values in a batch using the transformations registered in `ops`
    registered in `ops`: key, transformation`.
    The argument `element_wise` allows to process each value in the batch element wise.

    """

    _allows_update = False

    def __init__(self, ops: Dict[str, Callable], element_wise: bool = False, **kwargs):
        """

        Parameters
        ----------
        ops
            A dictionary mapping keys to transformation functions applied
        element_wise
            If True, apply the transformation to each element in the batch.
        kwargs
            Other Parameters
        """
        super().__init__(**kwargs)
        self.ops = ops
        self.element_wise = element_wise

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        for key, op in self.ops.items():
            values = batch[key]
            if self.element_wise:
                batch[key] = [op(x) for x in values]
            else:
                batch[key] = op(values)

        return batch


class ApplyToAll(Pipe):
    """
    Apply a transformation
    registered in `ops`: key, transformation`.
    The argument `element_wise` allows to process each value in the batch element wise.
    """

    _allows_update = False

    def __init__(
        self,
        op: Callable,
        element_wise: bool = False,
        allow_kwargs: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        op
            The transformation function applied to each batch value
        element_wise
            If True, apply the transformation to each element in the batch.
        allow_kwargs
            If True, the transformation function can take keyword arguments.
        kwargs
            Other Parameters
        """
        super().__init__(**kwargs)
        self.op = op
        self.element_wise = element_wise
        self.allow_kwargs = allow_kwargs

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        if not self.allow_kwargs:
            kwargs = {}
        for key, values in batch.items():
            if self.element_wise:
                batch[key] = [self.op(x, **kwargs) for x in values]
            else:
                batch[key] = self.op(values, **kwargs)
        return batch


class CopyBatch(Pipe):
    """Copy an input batch"""

    _allows_update = False
    _allows_input_filter = False

    def __init__(self, *, deep: bool = False, **kwargs):
        """

        Parameters
        ----------
        deep
            If True, copy the input batch recursively (deepcopy)
        kwargs
            Other Parameters
        """
        super().__init__(**kwargs)
        self.deep = deep

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        if self.deep:
            return deepcopy(batch)
        else:
            return copy(batch)
