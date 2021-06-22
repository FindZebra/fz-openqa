import torch
from torch import Tensor

from .base import Similarity


class MaxSim(Similarity):
    def __init__(self, similarity_metric=str):
        assert similarity_metric in {"l2", "dot_product"}
        self.similarity_metric = similarity_metric

    def __call__(self, query: Tensor, document: Tensor) -> Tensor:
        """Compute the similarity between a batch of N queries and
        M elements. Returns a similarity matrix S of shape N x M
        where S_ij is the similarity between the query[i] and document[i]
        `query` and `document` are global representation (shape [bs, h]).
        If `query` or `document` is tensor of dimension 3, take the
        first element (corresponds to the CLS token)"""
        assert len(document.shape) == 3
        assert len(query.shape) == 3
        n = query.shape[0]
        m = document.shape[0]

        if self.similarity_metric == "dot_product":
            # todo: check how this is implemented in the official repo,
            #  they do not return an NxM matrix
            interactions = torch.einsum("nph, mqh -> nmpq", query, document)
            sim = interactions.max(dim=-1).values.sum(-1)
        elif self.similarity_metric == "l2":
            raise NotImplementedError
        else:
            raise ValueError(
                f"Unknown similarity metric: `{self.similarity_metric}`"
            )

        assert n == sim.shape[0]
        assert m == sim.shape[1]
        return sim
