from unittest import TestCase

import numpy as np

from fz_openqa.datamodules.index.search_result import SearchResult


class TestSearchResult(TestCase):

    def test_fill_negative_indexes(self):
        # check that ngative indexes are replaced with random indexes
        dset_size = 10
        search_results = SearchResult(index=1000 * [[1, -1, -1]],
                                      score=1000 * [[1., 1., 1.]], dataset_size=dset_size)
        index = np.array(search_results.index)
        self.assertTrue((index >= 0).all())
        self.assertTrue((index < dset_size).all())
