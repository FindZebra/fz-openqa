from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.pretty import pprint_batch


def always_true(*args, **kwargs):
    return True


class Pipe:
    """
    A pipe is a small unit of computation that ingests,
    modify and returns a batch of data.
    """

    @staticmethod
    def eg(batch: Batch, idx: int, filter_op: Optional[Callable] = None):
        """Extract example `idx` from a batch, potentially filter the keys"""
        filter_op = filter_op or always_true
        return {k: v[idx] for k, v in batch.items() if filter_op(k)}

    def batch_size(self, batch: Batch) -> int:
        return len(next(iter(batch.values())))

    def __call__(self, batch: Union[List[Batch], Batch], **kwargs) -> Batch:
        """The call of the pipeline process"""
        raise NotImplementedError


class Identity(Pipe):
    """
    A pipe that passes a batch without modifying it.
    """

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        return batch


class Lambda(Pipe):
    """
    Apply a lambda function to the batch.
    """

    def __init__(self, op: Callable):
        self.op = op

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        return self.op(batch)


class FilterKeys(Pipe):
    """
    Filter the keys in the batch.
    """

    def __init__(self, condition: Callable):
        self.condition = condition

    def __call__(
        self, batch: Union[List[Batch], Batch], **kwargs
    ) -> Union[List[Batch], Batch]:
        """The call of the pipeline process"""
        return self.filter(batch)

    def filter(self, batch):
        return {k: v for k, v in batch.items() if self.condition(k)}


class DropKeys(Pipe):
    """
    Filter the keys in the batch.
    """

    def __init__(self, keys: List[str]):
        self.keys = keys

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        for key in self.keys:
            batch.pop(key)
        return batch


class AddPrefix(Pipe):
    """
    Append the keys with a prefix.
    """

    def __init__(self, prefix: str, **kwargs):
        self.prefix = prefix

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        return {f"{self.prefix}{k}": v for k, v in batch.items()}


class ReplaceInKeys(Pipe):
    """
    Remove a the prefix in each key.
    """

    def __init__(self, a: str, b: str, **kwargs):
        self.a = a
        self.b = b

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        return {k.replace(self.a, self.b): v for k, v in batch.items()}


class Rename(Pipe):
    """
    Rename a set of keys
    """

    def __init__(self, keys: Dict[str, str], **kwargs):
        self.keys = keys

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        for old_key, new_key in self.keys.items():
            value = batch.pop(old_key)
            batch[new_key] = value

        return batch


class Apply(Pipe):
    """
    Transform the values in a batch for all transformations
    registered in `ops`: key, transformation`.
    The argument `element_wise` allows to process each value in the batch element wise.
    """

    def __init__(self, ops: Dict[str, Callable], element_wise: bool = False):
        self.ops = ops
        self.element_wise = element_wise

    def __call__(self, batch: Batch, **kwargs) -> Batch:
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

    def __init__(self, op: Callable, element_wise: bool = False):
        self.op = op
        self.element_wise = element_wise

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        for key, values in batch.items():
            if self.element_wise:
                batch[key] = [self.op(x) for x in values]
            else:
                batch[key] = self.op(values)

        return batch


class PrintBatch(Pipe):
    """
    Print the batch
    """

    def __init__(self, header: Optional[str] = None):
        self.header = header

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        """The call of the pipeline process"""
        pprint_batch(batch, header=self.header)

        return batch
