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


def cast_to_numpy(x: Any, as_contiguous: bool = True, dtype: Optional[str] = None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().to(device="cpu").numpy()
        if dtype is not None:
            x = x.astype(dtype)
    elif isinstance(x, np.ndarray):
        pass
    else:
        x = np.array(x)

    if as_contiguous:
        x = np.ascontiguousarray(x)

    return x


def cast_to_torch(
    x: Any,
    as_contiguous: bool = True,
    dtype: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        pass
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        x = x.to(device=device, dtype=dtype)
    else:
        x = torch.tensor(x, device=device, dtype=dtype)

    if as_contiguous:
        x = x.contiguous()

    return x.to(device=device)


def cast_values_to_numpy(
    batch: Batch, as_contiguous: bool = False, dtype: Optional[str] = None
) -> Batch:
    return {k: cast_to_numpy(v, as_contiguous=as_contiguous, dtype=dtype) for k, v in batch.items()}


def cast_values_to_torch(
    batch: Batch,
    as_contiguous: bool = True,
    dtype: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> Batch:
    return {
        k: cast_to_torch(v, as_contiguous=as_contiguous, dtype=dtype, device=device)
        for k, v in batch.items()
    }


def always_true(*args, **kwargs):
    return True


def get_batch_eg(batch: Batch, idx: int, filter_op: Optional[Callable] = None) -> Dict:
    """Extract example `idx` from a batch, potentially filter the keys"""
    filter_op = filter_op or always_true
    return {k: v[idx] for k, v in batch.items() if filter_op(k)}


def infer_batch_size(batch: Batch) -> int:
    batch = {k: v for k, v in batch.items() if v is not None and not isinstance(v, (Number, str))}
    bss = [len(v) for v in batch.values()]
    bs = bss[0]
    if not all(bs == bs_ for bs_ in bss):
        lengths = ", ".join([f"{k}={len(v)}" for k, v in batch.items()])
        raise ValueError(
            f"Fields are not of the same length. Cannot infer batch size. Lengths=({lengths})"
        )
    return bs


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
    r = x == y
    if isinstance(r, np.ndarray):
        return r.all()
    elif isinstance(r, bool):
        return r
    else:
        raise TypeError(f"Cannot check equality of {type(r)}")


def iter_batch_rows(batch: Batch) -> Iterable[Dict]:
    """iterate through each batch example"""
    batch_size = infer_batch_size(batch)
    for i in range(batch_size):
        yield get_batch_eg(batch, idx=i)


def is_index_contiguous(indexes):
    """Check if indexes are contiguous: i.e I[i+1] = I[i] + 1"""
    return all(p + 1 == n for p, n in zip(indexes[:-1], indexes[1:]))
