from unittest import TestCase
import torch
import numpy as np
from parameterized import parameterized

from fz_openqa.datamodules.index.search_result import SearchResult


class TestSearchResult(TestCase):

    @parameterized.expand([
        (lambda x:x,),
        (torch.tensor,),
        (np.array,),
    ])
    def test_fill_negative_indexes(self, op):
        # check that ngative indexes are replaced with random indexes
        dset_size = 10
        search_results = SearchResult(index=op(1000 * [[1, -1, -1]]),
                                      score=op(1000 * [[1., 1., 1.]]),
                                      dataset_size=dset_size,
                                      k=3)
        index = np.array(search_results.index)
        self.assertTrue((index >= 0).all())
        self.assertTrue((index < dset_size).all())
