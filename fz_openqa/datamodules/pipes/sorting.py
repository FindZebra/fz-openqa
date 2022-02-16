from typing import Any
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from torch import Tensor

from .base import Pipe
from fz_openqa.utils.datastruct import Batch


def reindex(x: Any, index: Union[np.ndarray, Tensor, List[int]]) -> Any:
    if isinstance(x, (Tensor, np.ndarray)):
        return x[index]
    elif isinstance(x, (list, tuple)):
        return [x[i] for i in index]
    else:
        raise ValueError(f"Cannot handle type {type(x).__name__}")


class Sort(Pipe):
    """Sort a batch according to some key values"""

    def __init__(
        self,
        keys: List[str],
        *,
        reverse: bool = True,
        **kwargs,
    ):
        super(Sort, self).__init__(**kwargs)
        self.keys = keys
        self.reverse = reverse

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        self._check_input_keys(batch)

        # get values and index
        values = zip(*(batch[key] for key in self.keys))
        indexed_values = [(i, values) for i, values in enumerate(values)]

        # sort the index
        def _key(u: Tuple) -> Tuple:
            i, v = u
            return v

        indexed_values = sorted(indexed_values, key=_key, reverse=self.reverse)
        index = [i for i, _ in indexed_values]

        return {k: reindex(v, index) for k, v in batch.items()}

    def _check_input_keys(self, batch):
        for key in self.keys:
            assert key in batch.keys(), f"key={key} not in batch with keys={list(batch.keys())}"
