from typing import Any
from typing import Callable
from typing import List

from fz_openqa.utils.datastruct import Batch


class Condition:
    """
    This class implements a condition for the control pipe.
    """

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

    def todict(self) -> dict:
        """
        Returns a dictionary representation of the condition.

        Returns
        -------
        Dict
            Dictionary representation of the condition.
        """
        return {"__type__": type(self).__name__, **vars(self)}

    def __repr__(self):
        """
        Returns a string representation of the condition.

        Returns
        -------
        str
            String representation of the condition.
        """
        args = [f"{k}={v}" for k, v in self.todict().items()]
        return self.__class__.__name__ + "(" + ", ".join(args) + ")"


class Contains(Condition):
    """check if the key is in the set of `allowed_keys`"""

    def __init__(self, pattern: str):
        self.pattern = pattern

    def __call__(self, x: Any, **kwargs) -> bool:
        return self.pattern in x


class In(Condition):
    """check if the key is in the set of `allowed_keys`"""

    def __init__(self, allowed_values: List[str]):
        self.allowed_keys = allowed_values

    def __call__(self, x: Any, **kwargs) -> bool:
        return x in self.allowed_keys


class HasPrefix(Condition):
    """check if the key starts with a given prefix"""

    def __init__(self, prefix: str):
        self.prefix = prefix

    def __call__(self, key: Any, **kwargs) -> bool:
        return str(key).startswith(self.prefix)


class Reduce(Condition):
    """
    Reduce multiple conditions into outcome.
    """

    def __init__(self, *conditions: Condition, reduce_op: Callable = all):
        self.reduce_op = reduce_op
        self.conditions = list(conditions)

    def __call__(self, batch: Batch, **kwargs) -> bool:
        return self.reduce_op(c(batch) for c in self.conditions)

    def __repr__(self):
        return f"{self.__class__.__name__}(conditions={list(self.conditions)}, op={self.reduce_op})"


class Not(Condition):
    """`not` Operator for a condition."""

    def __init__(self, condition: Condition):
        self.condition = condition

    def __call__(self, batch: Batch, **kwargs) -> bool:
        return not self.condition(batch)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.condition})"


class Static(Condition):
    """Condition with a static boolean outcome."""

    def __init__(self, cond: bool):
        self.cond = cond

    def __call__(self, batch: Batch, **kwargs) -> bool:
        return self.cond

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cond})"
