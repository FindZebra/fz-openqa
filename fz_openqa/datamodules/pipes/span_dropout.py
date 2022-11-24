from typing import List
from typing import Optional

import numpy as np
import torch
from datasets import Split
from warp_pipes import Batch
from warp_pipes import Pipe
from warp_pipes.core.condition import In


class SpanDropout(Pipe):
    """
    Chunk the input into spans of length `span_length~Poisson(lbda)` and
    drop each span according to probability `rate`.
    """

    def __init__(
        self,
        lbda: int = 3,
        rate: float = 0.2,
        field: str = "question",
        keys: List[str] = None,
        input_filter=None,
        **kwargs,
    ):
        keys = keys or ["input_ids", "attention_mask"]
        self.keys = [f"{field}.{key}" for key in keys]
        self.lbda = lbda
        self.rate = rate
        assert input_filter is None, "SpanMasking does not support input_filter"
        input_filter = In(self.keys)
        super(SpanDropout, self).__init__(**kwargs, input_filter=input_filter)

    def _call_batch(
        self, batch: Batch, idx: Optional[List[int]] = None, split: Optional[Split] = None, **kwargs
    ) -> Batch:

        if self.rate == 0:
            return {}

        if split != Split.TRAIN:
            return {}

        first_key = self.keys[0]
        x = batch[first_key]
        length = x.shape[-1]
        idx = 0
        output = {k: None for k in self.keys}
        while idx < length:
            step_size = int(np.random.poisson(self.lbda))
            drop_it = np.random.random() < self.rate

            if not drop_it:
                output = {
                    k: self._cat(v, batch[k][..., idx : idx + step_size]) for k, v in output.items()
                }

            idx += step_size

        if output[first_key] is None:
            return {}

        return output

    def _cat(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a is None:
            return b
        return torch.cat((a, b), dim=-1)
