from torch import Tensor


class Similarity():
    def __call__(self, query: Tensor, x: Tensor) -> Tensor:
        """Compute the similarity between a batch of N queries and
        M x elements. Returns a similarity matrix of shape N x M"""
        raise NotImplementedError
