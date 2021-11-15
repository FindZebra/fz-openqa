import abc
from functools import singledispatchmethod
from typing import List

from fz_openqa.datamodules.pipes.control.condition import Condition
from fz_openqa.utils.datastruct import Batch


class BatchCondition(Condition):
    """
    Condition operating on the batch level.
    """

    __metaclass__ = abc.ABCMeta

    @singledispatchmethod
    def __call__(self, batch: Batch) -> bool:
        raise TypeError(f"Cannot handle input of type type {type(batch)}.")

    @__call__.register(dict)
    def _(self, batch: Batch) -> bool:
        return self._call_batch(batch)

    @__call__.register(list)
    def _(self, batch: List[Batch]) -> bool:
        return self._call_egs(batch)

    @abc.abstractmethod
    def _call_batch(self, batch: Batch, **kwargs) -> bool:
        raise NotImplementedError

    def _call_egs(self, batch: list) -> bool:
        first_eg = batch[0]
        return self._call_batch(first_eg)


class HasKeyWithPrefix(BatchCondition):
    """Test if the batch contains at least one key with the specified prefix"""

    def __init__(self, prefix: str, **kwargs):
        super().__init__(**kwargs)
        self.prefix = prefix

    def _call_batch(self, batch: Batch, **kwargs) -> bool:
        return any(str(k).startswith(self.prefix) for k in batch.keys())

    def __repr__(self):
        return f"{self.__class__.__name__}({self.prefix})"


class HasKeys(BatchCondition):
    """Test if the batch contains all the required keys"""

    def __init__(self, keys: List[str], **kwargs):
        super().__init__(**kwargs)
        self.keys = keys

    def _call_batch(self, batch: Batch, **kwargs) -> bool:
        return all(key in batch for key in self.keys)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.keys})"


class AllValuesOfType(BatchCondition):
    """Check if all batch values are of the specified type"""

    def __init__(self, cls: type, **kwargs):
        super().__init__(**kwargs)
        self.cls = cls

    def _call_batch(self, batch: Batch, **kwargs) -> bool:
        return all(isinstance(v, self.cls) for v in batch.values())

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cls.__name__})"
