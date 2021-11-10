from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

from . import Pipe
from ...utils.datastruct import Batch


class Collate(Pipe):
    """
    Create a Batch object from a list of examples, where an
    example is defined as a batch of one element.

    This default class concatenate values as lists.
    """

    def __init__(
        self,
        keys: Optional[Union[str, List[str], Callable]] = None,
        key_op: Optional[Callable] = None,
    ):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.key_op = key_op

    def __call__(self, examples: Union[Batch, Iterable[Batch]], **kwargs) -> Batch:
        # cast, filter keys and check type and keys consistency
        if isinstance(examples, dict):
            keys = self.get_keys_form_eg(examples)
            batch = {key: examples[key] for key in keys}
        else:
            examples = list(examples)
            first_eg = examples[0]
            keys = self.get_keys_form_eg(first_eg)
            self.check_consistency(examples, keys)

            # build a batch: {key: [values]}
            batch = {key: [eg[key] for eg in examples] for key in keys}

        # apply the operator
        if self.key_op is not None:
            batch = {self.key_op(k): v for k, v in batch.items()}

        return batch

    @staticmethod
    def check_consistency(examples, keys):
        assert all(isinstance(eg, dict) for eg in examples)
        assert all(keys.issubset(set(eg.keys())) for eg in examples)

    def get_keys_form_eg(self, first_eg):
        keys = set(first_eg.keys())
        if self.keys is not None:
            if isinstance(self.keys, (list, set, tuple)):
                _keys = set(self.keys)
            elif isinstance(self.keys, Callable):
                _keys = set(filter(self.keys, keys))
            else:
                raise TypeError(f"keys must be a list, set or callable, not {type(self.keys)}")

            keys = set.intersection(keys, _keys)
        return keys

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return self.keys or input_keys


class DeCollate(Pipe):
    def __call__(self, batch: Batch, **kwargs) -> List[Dict[str, Any]]:
        keys = list(batch.keys())
        length = len(batch[keys[0]])
        lengths = {k: len(v) for k, v in batch.items()}
        assert all(length == eg_l for eg_l in lengths.values()), f"un-equal lengths: {lengths}"
        return [{k: batch[k][i] for k in keys} for i in range(length)]


class FirstEg(Pipe):
    def __call__(self, examples: List[Batch]) -> Batch:
        return examples[0]


class ApplyToEachExample(Pipe):
    def __init__(self, pipe: Pipe):
        self.pipe = pipe

    def __call__(self, examples: Iterable[Batch]) -> Iterable[Batch]:
        for eg in examples:
            yield self.pipe(eg)
