from copy import deepcopy
from unittest import TestCase

import numpy as np
import rich
import torch

from fz_openqa.datamodules.pipes import Sort
from warp_pipes.core.condition import In
from warp_pipes import Batch


class TestSort(TestCase):
    def test_sort_batch_dtypes(self):
        """test sorting for list, np.ndarray and tensors"""
        batch = {"a": [1, 2, 3, 4], "b": [4, 3, 2, 1], "c": [2, 5, 7, 3]}

        assert self.check_if_values_are_the_same_length(batch)
        for cls in [lambda x: x, torch.tensor, np.array]:
            for reverse in [False, True]:
                self._test_sort_batch(
                    {k: cls(v) for k, v in deepcopy(batch).items()}, reverse=reverse
                )

    def check_if_values_are_the_same_length(self, batch):
        v0 = next(iter(batch.values()))
        same_length = all(len(x) == len(v0) for x in batch.values())
        return same_length

    def _test_sort_batch(self, batch: Batch, reverse=True):
        # init the pipe
        pipe = Sort(keys=["a"] if reverse else ["b"], reverse=reverse)

        # sort the batch
        reversed_batch = pipe(deepcopy(batch))

        # test that all keys are present
        self.assertSetEqual(set(batch.keys()), set(reversed_batch.keys()))

        # test that values are sorted
        self.assertSequenceEqual([u for u in batch["a"]], [u for u in reversed_batch["b"]])
        self.assertSequenceEqual([u for u in batch["b"]], [u for u in reversed_batch["a"]])

        # test that all values still have tyhe same length
        self.assertTrue(self.check_if_values_are_the_same_length(batch))

    def test_sort_with_filter(self):
        batch = {"a": [1, 2, 3, 4], "b": [4, 3, 2, 1], "c": [2, 5, 7, 3]}

        pipe = Sort(keys=["a"], reverse=True, input_filter=In(["a", "b"]), update=True)

        # sort the batch
        reversed_batch = pipe(deepcopy(batch))

        # check that c is NOT sorted
        self.assertSequenceEqual(reversed_batch["c"], batch["c"])

    def test_sort_multiple_keys(self):
        batch = {"a": [0, 0, 1, 1, 2, 2], "b": [5, 6, 3, 7, 1, 0], '_index_': [1, 2, 3, 4, 5, 6]}
        pipe = Sort(keys=["a", "b"], reverse=True)
        output = pipe(batch)
        self.assertEqual(output['_index_'], [5, 6, 4, 3, 2, 1])
