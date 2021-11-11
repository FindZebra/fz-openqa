from copy import copy
from copy import deepcopy
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from .base import Pipe
from fz_openqa.utils.datastruct import Batch


class Identity(Pipe):
    """
    A pipe that passes a batch without modifying it.
    """

    def _call(self, batch: Batch, **kwargs) -> Batch:
        """
        Returns the batch without modifying it.

        Parameters
        ----------
        batch
        kwargs

        Returns
        -------

        """
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

    def _call(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        return self.op(batch)

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return self._output_keys or super().output_keys(input_keys)


class GetKey(Pipe):
    def __init__(self, key: str, **kwargs):
        super().__init__(**kwargs)
        self.key = key

    def _call(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        return {self.key: batch[self.key]}

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return [self.key]


class FilterKeys(Pipe):
    """
    Filter the keys in the batch.
    """

    def __init__(self, condition: Optional[Callable], **kwargs):
        super().__init__(**kwargs)
        self.condition = condition

    def _call(self, batch: Union[List[Batch], Batch], **kwargs) -> Union[List[Batch], Batch]:
        """The call of the pipeline process"""
        if self.condition is None:
            return batch
        return self.filter(batch)

    def filter(self, batch):
        return {k: v for k, v in batch.items() if self.condition(k)}

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return [k for k in input_keys if self.condition(k)]


class DropKeys(Pipe):
    """
    Filter the keys in the batch.
    """

    def __init__(self, keys: List[str], **kwargs):
        super().__init__(**kwargs)
        self.keys = keys

    def _call(self, batch: Batch, **kwargs) -> Batch:
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

    def __init__(self, prefix: str, **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix

    def _call(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        return {f"{self.prefix}{k}": v for k, v in batch.items()}

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return [f"{self.prefix}{k}" for k in input_keys]


class ReplaceInKeys(Pipe):
    """
    Remove a the prefix in each key.
    """

    def __init__(self, a: str, b: str, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def _call(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        return {k.replace(self.a, self.b): v for k, v in batch.items()}

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return [k.replace(self.a, self.b) for k in input_keys]


class RenameKeys(Pipe):
    """
    Rename a set of keys
    """

    def __init__(self, keys: Dict[str, str], **kwargs):
        super().__init__(**kwargs)
        self.keys = keys

    def _call(self, batch: Batch, **kwargs) -> Batch:
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
    Transform the values in a batch for all transformations
    registered in `ops`: key, transformation`.
    The argument `element_wise` allows to process each value in the batch element wise.
    """

    def __init__(self, ops: Dict[str, Callable], element_wise: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.ops = ops
        self.element_wise = element_wise

    def _call(self, batch: Batch, **kwargs) -> Batch:
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
    Transform the values in a batch for all transformations
    registered in `ops`: key, transformation`.
    The argument `element_wise` allows to process each value in the batch element wise.
    """

    def __init__(
        self,
        op: Callable,
        element_wise: bool = False,
        allow_kwargs: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.op = op
        self.element_wise = element_wise
        self.allow_kwargs = allow_kwargs

    def _call(self, batch: Batch, **kwargs) -> Batch:
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
    def __init__(self, *, deep: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.deep = deep

    def _call(self, batch: Batch, **kwargs) -> Batch:
        if self.deep:
            return deepcopy(batch)
        else:
            return copy(batch)
