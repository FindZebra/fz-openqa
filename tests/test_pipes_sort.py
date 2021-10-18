from copy import deepcopy
from unittest import TestCase

import numpy as np
import torch

from fz_openqa.datamodules.pipes import Sort
from fz_openqa.utils.datastruct import Batch


class TestSort(TestCase):

    def test_sort_batch_dtypes(self):
        """test sorting for list, np.ndarray and tensors"""
        batch = {'a': [1, 2, 3, 4],
                 'b': [4, 3, 2, 1],
                 'c': [2, 5, 7, 3]}

        assert self.check_if_values_are_the_same_length(batch)
        for cls in [lambda x: x, torch.tensor, np.array]:
            for reversed in [False, True]:
                self._test_sort_batch(
                    {k: cls(v) for k, v in deepcopy(batch).items()},
                    reversed=reversed)

    def check_if_values_are_the_same_length(self, batch):
        v0 = next(iter(batch.values()))
        same_length = all(len(x) == len(v0) for x in batch.values())
        return same_length

    def _test_sort_batch(self, batch: Batch, reversed=True):
        # init the pipe
        pipe = Sort(key="a" if reversed else "b", reversed=reversed)

        # sort the batch
        reversed_batch = pipe(deepcopy(batch))

        # test that all keys are present
        self.assertSetEqual(set(batch.keys()), set(reversed_batch.keys()))

        # test that values are sorted
        self.assertSequenceEqual([u for u in batch['a']],
                                 [u for u in reversed_batch['b']])
        self.assertSequenceEqual([u for u in batch['b']],
                                 [u for u in reversed_batch['a']])

        # test that all values still have tyhe same length
        self.assertTrue(self.check_if_values_are_the_same_length(batch))

    def test_sort_with_filter(self):
        batch = {'a': [1, 2, 3, 4],
                 'b': [4, 3, 2, 1],
                 'c': [2, 5, 7, 3]}

        pipe = Sort(key="a", reversed=True,
                    filter=lambda key: key in {"a", "b"})

        # sort the batch
        reversed_batch = pipe(deepcopy(batch))

        # check that c is NOT sorted
        self.assertSequenceEqual(reversed_batch["c"], batch["c"])
