from unittest import TestCase

import numpy as np
from parameterized import parameterized

from fz_openqa.utils.shape import infer_missing_dims, infer_shape_nested_list, infer_batch_shape


class TestShape(TestCase):
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
        ([1, 2, 3, 4],),
        ([[1, 2, 3], [4, 5, 6]],),
        ([[[1, 2], [3, 4]], [[4, 5], [6, 7]]],),
        ([[1, 2, 3], [4, 5]],),
        ([[[1], 2], [4, 5]],),
    ])
    def test_infer_shape_nested_list(self, values):
        shape = infer_shape_nested_list(values)
        self.assertEqual(shape, list(np.array(values).shape))

    @parameterized.expand([
        ({'a': [1, 2], 'b': [[1, 2, 3], [1, 2]]}, [2]),
        ({'a': [[[1, 1], [1, 1]], [[2, 2], [2, 2]]],
          'b': [[[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]]},
         [2, 2, 2]),
        ({'document.text': [['jkhnds', 'aff', 'asdsd']],
          'document.input_ids': [[[1, 2], [1, 2, 3], [1, 2, 3, 4]]],
          'question.text': ['bajk'],
          'question.input_ids': [[1, 2, 3]]
          }, [1])
    ]
    )
    def test_infer_batch_shape(self, batch, expected_shape):
        self.assertEqual(infer_batch_shape(batch), expected_shape)
