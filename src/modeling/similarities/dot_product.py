from torch import Tensor

from .base import GolbalSimilarity


class DotProduct(GolbalSimilarity):
    def vec_similarity(self, x: Tensor, y: Tensor) -> Tensor:
        return x @ y.transpose(1, 0)
