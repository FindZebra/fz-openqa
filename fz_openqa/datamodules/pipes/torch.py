from typing import Any
from typing import Callable
from typing import Union

import torch
from pytorch_lightning.utilities import move_data_to_device
from torch import Tensor

from ...utils.functional import cast_values_to_numpy
from .base import Pipe
from fz_openqa.utils.datastruct import Batch


class ToNumpy(Pipe):
    """Move Tensors to the CPU and cast to numpy arrays."""

    def __init__(self, as_contiguous: bool = True, **kwargs):
        super(ToNumpy, self).__init__(**kwargs)
        self.as_contiguous = as_contiguous

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        return cast_values_to_numpy(batch, as_contiguous=self.as_contiguous)


class Itemize(Pipe):
    """Convert all values to lists."""

    def itemize(self, values: Any):
        if isinstance(values, Tensor) and values.dim() > 0:
            return [self.itemize(x) for x in values]
        elif isinstance(values, Tensor):
            return values.detach().item()
        else:
            return values

    def __call__(self, batch: Batch) -> Batch:
        return {k: self.itemize(v) for k, v in batch.items()}


class Forward(Pipe):
    """Process a batch of data using a model: output[key] = model(batch)"""

    def __init__(self, *, model: Union[Callable, torch.nn.Module], **kwargs):
        super(Forward, self).__init__()
        self.model = model

    @torch.no_grad()
    def __call__(self, batch: Batch, **kwargs) -> Batch:
        """Compute one batch of vectors"""

        # move data to device
        if isinstance(self.model, torch.nn.Module):
            device = next(iter(self.model.parameters())).device
            batch = move_data_to_device(batch, device)

        # process with the model (Dense or Sparse)
        return self.model(batch)
