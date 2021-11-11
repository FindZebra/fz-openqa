import json

import numpy as np
import rich
from parameterized import parameterized
from unittest import TestCase

from fz_openqa.datamodules.pipes.utils.nesting import flatten_nested, infer_missing_dims, \
    nested_list


class TestNested(TestCase):

    @parameterized.expand([
        ([[1, 2, 3], [4, 5, 6]], 1, [1, 2, 3, 4, 5, 6]),
        ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 1, [[1, 2], [3, 4], [5, 6], [7, 8]]),
        ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 2, [1, 2, 3, 4, 5, 6, 7, 8]),
    ])
    def test_flatten_nested(self, values, level, expected):
        output = list(flatten_nested(values, level=level))
        self.assertEqual(json.dumps(output), json.dumps(expected))


    @parameterized.expand([
        (64, [-1, 8], [8,8]),
        (64, [8, -1], [8, 8]),
        (128, [8, -1, 2], [8, 8, 2]),
        (128, [8, 2, -1], [8, 2, 8]),

    ])
    def test_infer_missing_dims(self, n_elements, shape, expected):
        new_shape = infer_missing_dims(n_elements=n_elements, shape=shape)
        self.assertEqual(np.prod(new_shape), n_elements)
        self.assertEqual(new_shape, expected)

    @parameterized.expand([
        (list(range(64)), (-1, 8)),
        (list(range(64)), (8, -1)),
        (list(range(64)), (8, -1, 2, 2)),
        (list(range(64)), (2, 2, 2, 2, -1)),
    ])
    def test_nested_list(self, values, shape):
        expected = np.array(values).reshape(shape)
        output = np.array(nested_list(values, shape=shape))
        self.assertTrue((expected==output).all())
