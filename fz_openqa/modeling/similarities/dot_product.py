import torch
from torch import Tensor

from .base import GolbalSimilarity


class DotProduct(GolbalSimilarity):
    def vec_similarity(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.einsum("nh, mh -> nm", x, y)
