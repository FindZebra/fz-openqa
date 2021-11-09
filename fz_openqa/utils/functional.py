from numbers import Number
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Union

import numpy as np
import torch.nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor

from fz_openqa.utils.datastruct import Batch


def is_loggable(x: Any):
    return (
        isinstance(x, Number)
        or (isinstance(x, Tensor) and x.numel() == 1)
        or (isinstance(x, np.ndarray) and len(x.reshape(-1)) == 1)
    )


def maybe_instantiate(conf_or_obj: Union[Any, DictConfig], **kwargs):
    if isinstance(conf_or_obj, (DictConfig, dict)):
        return instantiate(conf_or_obj, **kwargs)

    return conf_or_obj


def infer_device(model):
    return next(iter(model.parameters())).device


def only_trainable(parameters: Iterable[torch.nn.Parameter]):
    return (p for p in parameters if p.requires_grad)


def batch_reduce(x, op=torch.sum):
    return op(x.view(x.size(0), -1), dim=1)


def cast_to_numpy(x: Any, as_contiguous: bool = True) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().to(device="cpu").numpy()
    elif isinstance(x, np.ndarray):
        pass
    else:
        x = np.array(x)

    if as_contiguous:
        x = np.ascontiguousarray(x)

    return x


def cast_values_to_numpy(batch: Batch, as_contiguous: bool = True) -> Batch:
    return {k: cast_to_numpy(v, as_contiguous=as_contiguous) for k, v in batch.items()}


def always_true(*args, **kwargs):
    return True


def get_batch_eg(batch: Batch, idx: int, filter_op: Optional[Callable] = None) -> Dict:
    """Extract example `idx` from a batch, potentially filter the keys"""
    filter_op = filter_op or always_true
    return {k: v[idx] for k, v in batch.items() if filter_op(k)}


def infer_batch_size(batch: Batch) -> int:
    def _cond(k):
        return not k.startswith("__") and not k.endswith("__")

    return next(iter((len(v) for k, v in batch.items() if _cond(k))))


def infer_stride(batch: Batch) -> int:
    def _cond(k):
        return not k.startswith("__") and not k.endswith("__")

    x = next(iter(v for k, v in batch.items() if _cond(k)))
    stride = len(next(iter(x)))
    assert all(stride == len(y) for y in x)
    return stride


def check_equal_arrays(x, y):
    """check if x==y"""
    x = cast_to_numpy(x)
    y = cast_to_numpy(y)
    return (x == y).all()
