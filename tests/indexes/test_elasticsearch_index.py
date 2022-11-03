import unittest

from warp_pipes import Index, ElasticSearchIndex
from fz_openqa.datamodules.index.utils.es_engine import ping_es
from tests.indexes.test_base_index import TestIndex


@unittest.skipUnless(ping_es(), "Elastic Search is not reachable.")
class TestElasticSearchIndex(TestIndex):
    cls: Index.__class__ = ElasticSearchIndex

    def test_dill_inspect(self):
        self._test_dill_inspect()

    def test_search(self):
        self._test_search()
