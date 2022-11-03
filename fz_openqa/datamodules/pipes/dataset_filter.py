import abc
from typing import List
from typing import Optional

import numpy as np
from warp_pipes import Batch
from warp_pipes import Pipe
from warp_pipes.core.condition import In


class DatasetFilter(Pipe):
    _allows_update = False

    def __call__(self, *args, **kwargs) -> List[bool]:
        return super(DatasetFilter, self).__call__(*args, **kwargs)

    @abc.abstractmethod
    def _call_batch(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> List[bool]:
        ...


class SupervisedDatasetFilter(DatasetFilter):
    """Filter questions with no positive documents
    and (if max_positves is set) too many documents"""

    def __init__(
        self, *, key: str = "document.match_score", max_positives: Optional[int] = None, **kwargs
    ):
        input_filter = In([key])
        super().__init__(input_filter=input_filter, **kwargs)
        self.key = key
        self.max_positives = max_positives

    def _call_batch(self, batch: Batch, idx: Optional[List[int]] = None, **kwargs) -> List[bool]:
        score = np.array(batch[self.key])
        status = score.reshape(score.shape[0], -1).sum(axis=1) > 0

        if self.max_positives is not None:
            n_positive = (score > 0).sum(axis=-1)
            if len(n_positive.shape) > 1:
                n_positive = n_positive.reshape(n_positive.shape[0], -1)
                n_positive = n_positive.max(axis=1)

            status = status & (n_positive <= self.max_positives)

        return status
