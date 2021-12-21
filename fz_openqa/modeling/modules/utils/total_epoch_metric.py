from typing import Any
from typing import Optional

import torch
from torchmetrics import Metric


class TotalEpochMetric(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, values: torch.Tensor, *args, **kwargs) -> None:
        self.total += values.sum()

    def compute(self) -> Any:
        return self.total

    @property
    def is_differentiable(self) -> Optional[bool]:
        return False
