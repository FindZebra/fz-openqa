from typing import Callable
from typing import List
from typing import Optional
from typing import Union

from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch


class UpdateKeys(Pipe):
    """
    Update values of the keys in the batch.
    """

    def __init__(self, condition: Optional[Callable], **kwargs):
        super().__init__(**kwargs)
        self.condition = condition

    def __call__(self, batch: Union[List[Batch], Batch], **kwargs) -> Union[List[Batch], Batch]:
        """The call of the pipeline process"""
        if self.condition is None:
            return batch
        return self.update(batch)

    def update(self, batch):
        return {k: v.tolist() for k, v in batch.items() if self.condition(v)}

    def output_keys(self, input_keys: List[str]) -> List[str]:
        return [k for k in input_keys if self.condition(k)]