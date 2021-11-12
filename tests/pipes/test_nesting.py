import json
from copy import deepcopy
from unittest import TestCase

import rich
from parameterized import parameterized

from fz_openqa.datamodules.pipes import ApplyAsFlatten, Identity, Lambda
from fz_openqa.datamodules.pipes.control.condition import WithPrefix
from fz_openqa.datamodules.pipes.nesting import Nested
from fz_openqa.utils.datastruct import Batch


class TestAsFlatten(TestCase):
    cls = ApplyAsFlatten

    @parameterized.expand([
        ({'document.text': [['a', 'b', 'c'], ['d', 'e', 'f']], 'question': [1, 2]}, 1),
        ({'document.text': [[['a', 'b', 'c'], ['d', 'e', 'f']]], 'question': [[1, 2]]}, 2)
    ])
    def test_identity(self, data, level):
        """This process a batch using the Idenity on nested fields"""
        # with update
        pipe = self.cls(Identity(), update=True, input_filter=WithPrefix("document."), level=level)
        output = pipe(data)
        self.assertEqual(output, data)

        # no update
        ref = deepcopy(data)
        ref.pop('question')
        pipe = self.cls(Identity(), update=False, input_filter=WithPrefix("document."), level=level)
        output = pipe(data)
        self.assertEqual(output, ref)


class TestNested(TestAsFlatten):
    cls = Nested

    @parameterized.expand([
        ({'a': [[1, 2, 3], [1, 2, 3]], 'b': [[1, 2, 3], [1, 2, 3]]}, 1,
         {'a': [[1, 2], [1, 2]], 'b': [[1, 2], [1, 2]]}),
        ({'a': 3 * [[[1, 2, 3], [1, 2, 3]]], 'b': 3 * [[[1, 2, 3], [1, 2, 3]]]}, 2,
         {'a': 3 * [[[1, 2], [1, 2]]], 'b': 3 * [[[1, 2], [1, 2]]]}),
    ])
    def test_drop_values(self, input, level, expected):
        """Test Nested using a pipe that changes the nested batch size."""

        def drop_values(batch: Batch) -> Batch:
            """drop values >= 3"""

            def f(x):
                return x < 3

            return {k: list(filter(f, v)) for k, v in batch.items()}

        inner_pipe = Lambda(drop_values)
        pipe = self.cls(inner_pipe, level=level)
        output = pipe(input)
        self.assertEqual(json.dumps(expected, sort_keys=True), json.dumps(output, sort_keys=True))

    @parameterized.expand([
        ({'a': [[1, 2, 3], [1, 2, 3]], 'b': [[1, 2, 3], [1, 2, 3]]}, 1,
         {'a': [[3, 2, 1], [3, 2, 1]], 'b': [[3, 2, 1], [3, 2, 1]]}),
        ({'a': 3 * [[[1, 2, 3], [1, 2, 3]]], 'b': 3 * [[[1, 2, 3], [1, 2, 3]]]}, 2,
         {'a': 3 * [[[3, 2, 1], [3, 2, 1]]], 'b': 3 * [[[3, 2, 1], [3, 2, 1]]]}),
    ])
    def test_sort_values(self, input, level, expected):
        """Test Nested using a pipe that changes the nested batch size."""

        def sort_values(batch: Batch) -> Batch:
            """reverse sort values"""
            return {k: list(sorted(v, reverse=True)) for k, v in batch.items()}

        inner_pipe = Lambda(sort_values)
        pipe = self.cls(inner_pipe, level=level)
        output = pipe(input)
        self.assertEqual(json.dumps(expected, sort_keys=True), json.dumps(output, sort_keys=True))
