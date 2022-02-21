import logging
from copy import copy

from fz_openqa.datamodules.index.base import Index
from fz_openqa.datamodules.index.colbert import ColbertIndex
from fz_openqa.datamodules.index.dense import DenseIndex
from fz_openqa.datamodules.index.es import ElasticSearchIndex
from fz_openqa.datamodules.index.static import StaticIndex

logger = logging.getLogger(__name__)


class IndexBuilder:
    cls: Index.__class__ = Index

    def __init__(self, **kwargs):
        self.params = kwargs

    def __call__(self, **kwargs) -> Index:
        params = copy(self.params)
        params.update(kwargs)
        return self.cls(**params)


class FaissIndexBuilder(IndexBuilder):
    cls = DenseIndex


class ColbertIndexBuilder(IndexBuilder):
    cls = ColbertIndex


class ElasticSearchIndexBuilder(IndexBuilder):
    cls = ElasticSearchIndex


class StaticIndexBuilder(IndexBuilder):
    cls = StaticIndex


class FaissOrEsIndexBuilder(IndexBuilder):
    cls = DenseIndex
    alt_cls = ElasticSearchIndex

    def __call__(self, **kwargs) -> Index:
        params = copy(self.params)
        params.update(kwargs)

        if params.get("model", None) is not None:
            return self.cls(**params)
        else:
            logger.warning("No model specified, using ElasticSearchIndex")
            return self.alt_cls(**params)


class ColbertOrEsIndexBuilder(FaissOrEsIndexBuilder):
    cls = ColbertIndex
    alt_cls = ElasticSearchIndex
