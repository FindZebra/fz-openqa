from unittest import TestCase

from fz_openqa.datamodules.pipes.documents import select_values


class TestSelectValues(TestCase):
    def test_select_values(self):
        x = [1, 2, 3, 4, 5, 6]
        y = select_values(x, k=3)
        self.assertEqual(len(y), 3)
