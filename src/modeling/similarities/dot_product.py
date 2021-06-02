from torch import Tensor

from .base import Similarity


class DotProduct(Similarity):
    def __call__(self, query: Tensor, x: Tensor) -> Tensor:
        assert len(query.shape) == 2
        assert len(x.shape) == 2
        n = query.shape[0]
        m = x.shape[0]

        sim = n @ m.transpose(1, 0)
        assert n == sim.shape[0]
        assert m == sim.shape[1]
        return sim
