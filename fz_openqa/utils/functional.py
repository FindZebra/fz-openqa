from numbers import Number
from typing import Any
from typing import Iterable
from typing import Union

import numpy as np
import torch.nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor


def is_loggable(x: Any):
    return (
        isinstance(x, Number)
        or (isinstance(x, Tensor) and len(x.view(-1)) == 1)
        or (isinstance(x, np.ndarray) and len(x.reshape(-1)) == 1)
    )


def maybe_instantiate(conf_or_obj: Union[Any, DictConfig]):
    if isinstance(conf_or_obj, DictConfig):
        return instantiate(conf_or_obj)

    return conf_or_obj


def infer_device(model):
    return next(iter(model.parameters())).device


def only_trainable(parameters: Iterable[torch.nn.Parameter]):
    return (p for p in parameters if p.requires_grad)


def batch_reduce(x, op=torch.sum):
    return op(x.view(x.size(0), -1), dim=1)
