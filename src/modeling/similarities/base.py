import warnings

from torch import Tensor


class Similarity():
    def __call__(self, query: Tensor, document: Tensor) -> Tensor:
        """Compute the similarity between a batch of N queries and
        M elements. Returns a similarity matrix S of shape N x M
        where S_ij is the similarity between the query[i] and document[i]"""
        raise NotImplementedError


class GolbalSimilarity(Similarity):

    def vec_similarity(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError
        # example
        return x @ y.transpose(1, 0)

    def __call__(self, query: Tensor, document: Tensor) -> Tensor:
        """Compute the similarity between a batch of N queries and
        M elements. Returns a similarity matrix S of shape N x M
        where S_ij is the similarity between the query[i] and document[i]
        `query` and `document` are global representation (shape [bs, h]).
        If `query` or `document` is tensor of dimension 3, take the
        first element (corresponds to the CLS token)"""

        if len(query.shape) > 2:
            warnings.warn(f"`query` <shape={query.shape}>  is of dimension 3, "
                          f"selecting the 1st token (query = query[:, 0, :])")
            query = query[:, 0, :]
        if len(document.shape) > 2:
            warnings.warn(f"`document` <shape={document.shape}> is of dimension 3, "
                          f"selecting the 1st token (document = document[:, 0, :])")
            document = document[:, 0, :]

        n = query.shape[0]
        m = document.shape[0]
        sim = self.vec_similarity(query, document)
        assert n == sim.shape[0]
        assert m == sim.shape[1]
        return sim
