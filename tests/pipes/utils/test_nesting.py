import json
from unittest import TestCase

import numpy as np
from parameterized import parameterized

from fz_openqa.datamodules.pipes.utils.nesting import flatten_nested, nested_list


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
        (list(range(64)), (-1, 8)),
        (list(range(64)), (8, -1)),
        (list(range(64)), (8, -1, 2, 2)),
        (list(range(64)), (2, 2, 2, 2, -1)),
    ])
    def test_nested_list(self, values, shape):
        expected = np.array(values).reshape(shape)
        output = np.array(nested_list(values, shape=shape))
        self.assertTrue((expected == output).all())
