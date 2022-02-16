from typing import List
from typing import Optional

from .base import Pipe
from fz_openqa.utils.datastruct import Batch


class ConcatTextFields(Pipe):
    """
    Concatenate two text fields
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


class ExtractGoldAnswer(Pipe):
    """
    Extract the gold answer from the answer options given the target
    """

    def __init__(
        self,
        *,
        answer_field: str = "answer",
        options_key: str = "answer",
        target_key: str = "target",
        output_key: str = "gold",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.answer_field = answer_field
        self.options_key = options_key
        self.target_key = target_key
        self.output_key = output_key

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        options = batch[f"{self.answer_field}.{self.options_key}"]
        targets = batch[f"{self.answer_field}.{self.target_key}"]
        key = f"{self.answer_field}.{self.output_key}"
        return {key: [opts[t] for t, opts in zip(targets, options)]}
