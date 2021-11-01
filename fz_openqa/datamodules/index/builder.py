from copy import copy

from fz_openqa.datamodules.index.base import Index
from fz_openqa.datamodules.index.dense import FaissIndex
from fz_openqa.datamodules.index.es import ElasticSearchIndex


class IndexBuilder:
    cls: Index.__class__ = Index

    def __init__(self, **kwargs):
        self.params = kwargs

    def __call__(self, **kwargs) -> Index:
        params = copy(self.params)
        params.update(kwargs)
        return self.cls(**params)


class FaissIndexBuilder(IndexBuilder):
    cls = FaissIndex


class ElasticSearchIndexBuilder(IndexBuilder):
    cls = ElasticSearchIndex
