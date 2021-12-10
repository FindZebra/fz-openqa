from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
from torch import Tensor

from fz_openqa.utils.datastruct import OutputFormat

Array = Union[List[List], np.ndarray, Tensor]


def concat_arrays(*a: Array) -> Array:
    arr_type = type(a[0])
    assert all(isinstance(x, arr_type) for x in a)
    if arr_type == list:
        return sum(a, [])
    elif arr_type == np.ndarray:
        return np.concatenate(a, axis=0)
    elif arr_type == Tensor:
        return torch.cat(a, dim=0)
    else:
        raise TypeError(f"Unsupported type: {type(a[0])}")


class FormatArray:
    def __init__(self, output_format: OutputFormat, dtype: Optional[str] = None):
        self.output_format = output_format
        self.dtype = dtype

    def __call__(self, x: Array) -> Array:
        if self.output_format is None:
            pass
        elif self.output_format == OutputFormat.NUMPY:
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            else:
                x = np.array(x)
            if self.dtype is not None and np.isscalar(x):
                x = x.astype(self.dtype)
        elif self.output_format == OutputFormat.TORCH:
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            else:
                x = torch.tensor(x)
            if x.dtype in [torch.float16, torch.float32]:
                x = x.to(self.dtype)
        else:
            raise ValueError(
                f"Unsupported output_format: {self.output_format}." f"OutputFormat={OutputFormat}"
            )

        return x