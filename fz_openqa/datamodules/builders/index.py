from copy import copy

from datasets import Dataset
from warp_pipes import Index


class IndexBuilder:
    cls: Index.__class__ = Index

    def __init__(self, **kwargs):
        self.params = kwargs

    def __call__(self, corpus: Dataset, **kwargs) -> Index:
        params = copy(self.params)
        params.update(kwargs)
        return self.cls(corpus, **params)
