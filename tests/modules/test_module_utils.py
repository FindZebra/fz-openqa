import torch
from unittest import TestCase

from fz_openqa.modeling.modules.utils import check_only_first_doc_positive


class TestCheckOnlyFirstDocPositive(TestCase):
    def test_check_only_first_doc_positive(self):
        # positive example
        batch = {'document.match_score': torch.tensor([[1, 0, 0], [1, 0, 0]])}
        check_only_first_doc_positive(batch)

        # negative examples
        batch = {'document.match_score': torch.tensor([[1, 1, 0], [1, 0, 0]])}
        self.assertRaises(ValueError, lambda : check_only_first_doc_positive(batch))
        batch = {'document.match_score': torch.tensor([[0, 0, 0], [1, 0, 0]])}
        self.assertRaises(ValueError, lambda: check_only_first_doc_positive(batch))
        batch = {'document.match_score': torch.tensor([[1, 0, 0], [0, 1, 0]])}
        self.assertRaises(ValueError, lambda: check_only_first_doc_positive(batch))
