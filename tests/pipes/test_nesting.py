from copy import deepcopy
from unittest import TestCase

from fz_openqa.datamodules.pipes import ApplyAsFlatten, Identity
from fz_openqa.datamodules.pipes.control.condition import WithPrefix


class TestAsFlatten(TestCase):

    def setUp(self) -> None:
        self._data = {'document.text': [['a', 'b', 'c'], ['d', 'e', 'f']], 'question': [1, 2]}

    @property
    def data(self):
        return deepcopy(self._data)

    def test_identity(self):
        # with update
        pipe = ApplyAsFlatten(Identity(), update=True, input_filter=WithPrefix("document."))
        output = pipe(self.data)
        self.assertEqual(output, self.data)

        # no update
        pipe = ApplyAsFlatten(Identity(), update=False, input_filter=WithPrefix("document."))
        output = pipe(self.data)
        data = self.data
        data.pop('question')
        self.assertEqual(output, data)
