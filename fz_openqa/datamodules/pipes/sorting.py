from typing import Any
from typing import Callable
from typing import Optional

import numpy as np
from torch import Tensor

from .base import Pipe
from fz_openqa.utils.datastruct import Batch


def reindex(x: Any, index: np.ndarray) -> Any:
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
        self.filter = filter
        self.reversed = reversed

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        assert self.key in batch.keys()
        index = np.argsort(batch[self.key])
        if self.reversed:
            if isinstance(index, Tensor):
                index = index.flip(0)
            else:
                index = index[::-1]

        batch.update(
            {
                k: reindex(v, index)
                for k, v in batch.items()
                if self.filter is None or self.filter(k)
            }
        )
        return batch
