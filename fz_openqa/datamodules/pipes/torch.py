from typing import Callable
from typing import Union

import torch
from pytorch_lightning.utilities import move_data_to_device

from .base import Pipe
from fz_openqa.utils.datastruct import Batch


class ToNumpy(Pipe):
    """Move Tensors to the CPU and cast to numpy arrays."""

    def __call__(self, batch: Batch) -> Batch:
        return {
            k: v.to(device="cpu").numpy() if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }


class Forward(Pipe):
    """Process a batch of data using a model: output[key] = model(batch)"""

    def __init__(
        self, *, model: Union[Callable, torch.nn.Module], output_key: str
    ):
        self.model = model
        self.output_key = output_key

    @torch.no_grad()
    def __call__(self, batch: Batch, **kwargs) -> Batch:
        """Compute one batch of vectors"""

        # move data to device
        if isinstance(self.model, torch.nn.Module):
            device = next(iter(self.model.parameters())).device
            batch = move_data_to_device(batch, device)

        # process with the model (Dense or Sparse)
        batch[self.output_key] = self.model(batch)

        # cast to numpy and return
        return batch
