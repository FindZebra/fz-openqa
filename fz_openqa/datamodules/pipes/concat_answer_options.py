from typing import List

from .base import Pipe
from fz_openqa.utils.datastruct import Batch


class ConcatTextFields(Pipe):
    """
    Apply a lambda function to the batch.
    """

    def __init__(self, keys: List[str], *, new_key: str = "concatenated", **kwargs):
        super().__init__(**kwargs)
        self.fields = keys
        self.new_field = new_key

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        columns = [batch[field] for field in self.fields]
        new_column = []
        for u in zip(*columns):
            new_column += [" ".join(u)]
        output = {self.new_field: new_column}
        return output

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return [self.new_field]
