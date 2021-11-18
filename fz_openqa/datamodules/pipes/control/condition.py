import abc
from typing import Any
from typing import Callable
from typing import List

from fz_openqa.datamodules.component import Component
from fz_openqa.utils.datastruct import Batch


class Condition(Component):
    """
    This class implements a condition for the control pipe.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, x: Any, **kwargs) -> bool:
        """
        Returns True if the input matches the condition.

        Parameters
        ----------
        x
            object to be tested.

        Returns
        -------
        bool
            True if the input matches the condition.
        """
        raise NotImplementedError


class Contains(Condition):
    """check if the key is in the set of `allowed_keys`"""

    def __init__(self, pattern: str, **kwargs):
        super().__init__(**kwargs)
        self.pattern = pattern

    def __call__(self, x: Any, **kwargs) -> bool:
        return self.pattern in x


class In(Condition):
    """check if the key is in the set of `allowed_keys`"""

    def __init__(self, allowed_values: List[str], **kwargs):
        super(In, self).__init__(**kwargs)
        self.allowed_keys = allowed_values

    def __call__(self, x: Any, **kwargs) -> bool:
        return x in self.allowed_keys

    def __repr__(self):
        return f"{self.__class__.__name__}({self.allowed_keys})"


class HasPrefix(Condition):
    """check if the key starts with a given prefix"""

    def __init__(self, prefix: str, **kwargs):
        super(HasPrefix, self).__init__(**kwargs)
        self.prefix = prefix

    def __call__(self, key: Any, **kwargs) -> bool:
        return str(key).startswith(self.prefix)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.prefix})"


class Reduce(Condition):
    """
    Reduce multiple conditions into outcome.
    """

    def __init__(self, *conditions: Condition, reduce_op: Callable = all, **kwargs):
        super(Reduce, self).__init__(**kwargs)
        self.reduce_op = reduce_op
        self.conditions = list(conditions)

    def __call__(self, batch: Batch, **kwargs) -> bool:
        return self.reduce_op(c(batch) for c in self.conditions)

    def __repr__(self):
        return f"{self.__class__.__name__}(conditions={list(self.conditions)}, op={self.reduce_op})"


class Not(Condition):
    """`not` Operator for a condition."""

    def __init__(self, condition: Condition, **kwargs):
        super(Not, self).__init__(**kwargs)
        self.condition = condition

    def __call__(self, batch: Batch, **kwargs) -> bool:
        return not self.condition(batch)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.condition})"


class Static(Condition):
    """Condition with a static boolean outcome."""

    def __init__(self, cond: bool, **kwargs):
        super(Static, self).__init__(**kwargs)
        self.cond = cond

    def __call__(self, batch: Batch, **kwargs) -> bool:
        return self.cond

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cond})"
