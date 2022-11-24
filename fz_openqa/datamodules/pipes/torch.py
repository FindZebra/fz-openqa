from typing import Any
from typing import Callable
from typing import Union

import numpy as np
import torch
from pytorch_lightning.utilities import move_data_to_device
from torch import Tensor
from warp_pipes import ApplyToAll
from warp_pipes import Batch
from warp_pipes import Pipe

from fz_openqa.utils.functional import cast_values_to_numpy


class ToNumpy(Pipe):
    """Move Tensors to the CPU and cast to numpy arrays."""

    def __init__(self, as_contiguous: bool = True, **kwargs):
        super(ToNumpy, self).__init__(**kwargs)
        self.as_contiguous = as_contiguous

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        return cast_values_to_numpy(batch, as_contiguous=self.as_contiguous)


class ToList(ApplyToAll):
    """Move Tensors to the CPU and cast to numpy arrays."""

    def __init__(self):
        def op(x: Any):
            if isinstance(x, Tensor):
                return x.cpu().numpy()
            if isinstance(x, np.ndarray):
                x = x.tolist()
            if not isinstance(x, list):
                raise TypeError(f"Cannot convert {type(x)} to list")
            return x

        super().__init__(op=op)


class Itemize(Pipe):
    """Convert all values to lists."""

    def itemize(self, values: Any):
        if isinstance(values, Tensor) and values.dim() > 0:
            return [self.itemize(x) for x in values]
        elif isinstance(values, Tensor):
            return values.detach().item()
        else:
            return values

    def _call_batch(self, batch: Batch) -> Batch:
        return {k: self.itemize(v) for k, v in batch.items()}


class Forward(Pipe):
    """Process a batch of data using a model: output[key] = model(batch)"""

    def __init__(self, *, model: Union[Callable, torch.nn.Module], **kwargs):
        super(Forward, self).__init__()
        self.model = model

    @torch.no_grad()
    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        """Compute one batch of vectors"""

        # move data to device
        if isinstance(self.model, torch.nn.Module):
            device = next(iter(self.model.parameters())).device
            batch = move_data_to_device(batch, device)

        # process with the model (Dense or Sparse)
        return self.model(batch)
