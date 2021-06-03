import torch
from torch import Tensor

from .base import GolbalSimilarity


class L2Distance(GolbalSimilarity):
    def vec_similarity(self, x: Tensor, y: Tensor) -> Tensor:
        assert len(x.shape) == 2
        assert len(y.shape) == 2
        assert x.shape[1] == y.shape[1]
        fn = lambda x: x[
            None
        ].contiguous()  # .contiguous() is required to avoid bugs in the backward pass
        return torch.cdist(fn(x), fn(y), p=2).squeeze(0)
