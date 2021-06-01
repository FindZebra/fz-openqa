from typing import *

import torch
from torch import nn, Tensor


class Evaluator(nn.Module):
    """A base class to evaluate model's output given a model and a batch of data"""

    def forward(self, model: nn.Module, batch: Any, **kwargs: Any) -> Dict[str, Tensor]:
        """The forward pass handles the processing of the batch given the model,
           compute the loss and metrics, and returns all as a dictionary"""
        raise NotImplemented

        # example
        logits = model(batch)
        nll = torch.nn.functional.cross_entropy(logits, batch.labels)
        return {'loss': nll}
