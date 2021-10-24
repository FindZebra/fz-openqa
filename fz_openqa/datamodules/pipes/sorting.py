from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from torch import Tensor

from .base import always_true
from .base import Pipe
from fz_openqa.utils.datastruct import Batch


def reindex(x: Any, index: Union[np.ndarray, List[int]]) -> Any:
    if isinstance(x, (Tensor, np.ndarray)):
        x = x[index]
        return x
    elif isinstance(x, (list, tuple)):
        return [x[i] for i in index]
    else:
        raise ValueError(f"Cannot handle type {type(x).__name__}")


class Sort(Pipe):
    """Sort a batch according to some values"""

    def __init__(
        self,
        key: str,
        filter: Optional[Callable] = None,
        reversed: bool = True,
    ):
        self.key = key
        self.filter = filter or always_true
        self.reversed = reversed

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        assert (
            self.key in batch.keys()
        ), f"key={self.key} not in batch with keys={list(batch.keys())}"
        values = batch[self.key]
        index = sorted(
            range(len(values)), key=values.__getitem__, reverse=self.reversed
        )

        batch.update(
            {k: reindex(v, index) for k, v in batch.items() if self.filter(k)}
        )
        return batch
