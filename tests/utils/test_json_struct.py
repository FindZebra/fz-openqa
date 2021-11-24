import json
from unittest import TestCase

from parameterized import parameterized
from fz_openqa.utils.json_struct import apply_to_json_struct, flatten_json_struct, \
    reduce_json_struct


class TestJsonStruct(TestCase):

    @parameterized.expand([
        ({'a': 1, 'b': 2, 'c': 3}, {'a': 2, 'b': 3, 'c': 4}),
        ({'a': 1, 'b': [1, 2]}, {'a': 2, 'b': [2, 3]}),
        ({'a': {'b': 1, 'c': [1, 2, 3]}}, {'a': {'b': 2, 'c': [2, 3, 4]}})
    ])
    def test_apply_to_json_struct(self, data, expected):
        output = apply_to_json_struct(data, lambda x: x + 1)
        self.assertEqual(json.dumps(output, sort_keys=True), json.dumps(expected, sort_keys=True))

    @parameterized.expand([
        ({'a': 1, 'b': 2, 'c': 3}, [1, 2, 3]),
        ({'a': 1, 'b': [2, 3]}, [1, 2, 3]),
        ({'a': {'b': 1, 'c': [2, 3, 4]}}, [1, 2, 3, 4]),
        ({'a': {'b': 1, 'c': [2, 3, {'d': 4}]}}, [1, 2, 3, 4])
    ])
    def test_flatten_json_struct(self, data, expected):
        output = list(flatten_json_struct(data))
        self.assertIsInstance(output, list)
        self.assertEqual(len(output), len(expected))
        self.assertEqual(set(output), set(expected))

    @parameterized.expand([
        ({'a': True, 'b': True}, all, True),
        ({'a': True, 'b': True}, all, True),
        ({'a': True, 'b': False}, all, False),
        ({'a': True, 'b': False}, any, True),
        ({'a': 1, 'b': 2}, sum, 3),
        ({'a': 1, 'b': {'c': 2, 'd': [3, 4]}}, sum, 10),
    ])
    def test_flatten_json_struct(self, data, reduce_op, expected):
        output = reduce_json_struct(data, reduce_op)
        self.assertEqual(output, expected)
