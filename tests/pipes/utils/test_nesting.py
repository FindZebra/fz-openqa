import json
from unittest import TestCase

import numpy as np
from parameterized import parameterized

from fz_openqa.datamodules.pipes.utils.nesting import flatten_nested, infer_missing_dims, \
    nested_list
from fz_openqa.utils.shape import infer_shape_nested_list, infer_min_shape
from fz_openqa.utils.pretty import pprint_batch


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
        (64, [-1, 8], [8, 8]),
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
        self.assertTrue((expected == output).all())

    @parameterized.expand([
        ([1, 2, 3, 4],),
        ([[1, 2, 3], [4, 5, 6]],),
        ([[[1, 2], [3, 4]], [[4, 5], [6, 7]]],),
        ([[1, 2, 3], [4, 5]],),
        ([[[1], 2], [4, 5]],),
    ])
    def test_infer_nested_lengths(self, values):
        shape = infer_shape_nested_list(values)
        self.assertEqual(shape, list(np.array(values).shape))

    @parameterized.expand([
        ({'a': [1, 2], 'b': [[1, 2, 3], [1, 2]]}, [2]),
        ({'a': [[[1, 1], [1, 1]], [[2, 2], [2, 2]]],
          'b': [[[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]]}, [2,2]),
    ])
    def test_infer_min_shape(self, batch, expected_shape):
        pprint_batch(batch)
        self.assertEqual(infer_min_shape(batch), expected_shape)
