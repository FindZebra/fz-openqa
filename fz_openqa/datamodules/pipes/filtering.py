from typing import Callable

from .base import Pipe
from .nesting import reconcat
from fz_openqa.utils.datastruct import Batch


class FilterExamples(Pipe):
    """Filter examples from batch."""

    def __init__(self, condition: Callable):
        self.condition = condition

    def __call__(self, batch: Batch, **kwargs) -> Batch:

        exs = []
        for i in range(self.batch_size(batch)):
            eg = self.get_eg(batch, i)
            if self.condition(eg, **kwargs):
                exs += [eg]

        if len(exs) == 0:
            return {}

        types = {k: type(v) for k, v in batch.items()}
        for key in batch.keys():
            values = [eg[key] for eg in exs]
            values = reconcat(values, types[key])

            batch[key] = values

        return batch
