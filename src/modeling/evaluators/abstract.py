import collections
from typing import *

import torch
from torch import nn, Tensor


class Evaluator(nn.Module):
    """A base class to evaluate model's output given a model and a batch of data. Track and compute metrics"""

    def forward(self, model: nn.Module, batch: Any, split: str, **kwargs: Any) -> Dict[str, Tensor]:
        """The forward pass handles the processing of the batch given the model,
        compute the loss and update the metrics. Return a dictionary output with at least the key 'loss'"""
        raise NotImplemented

        # example
        logits = model(batch)
        nll = torch.nn.functional.cross_entropy(logits, batch.labels)
        return {"loss": nll}

    def reset_metrics(self, split: Optional[str] = None) -> None:
        """reset the metrics"""
        raise NotImplementedError

    def compute_metrics(self, split: Optional[str] = None) -> Dict[str, Tensor]:
        """Compute the metrics"""
        raise NotImplemented

    def check_batch_type(self, batch):
        assert isinstance(
            batch,
            (
                dict,
                collections.OrderedDict,
                collections.UserDict,
            ),
        )
