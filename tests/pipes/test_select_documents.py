from copy import deepcopy
from unittest import TestCase

import torch

from fz_openqa.datamodules.pipes.documents import SelectDocsOneEg


class TestSelectDocuments(TestCase):

    def setUp(self) -> None:
        self.data = {'document.feature': torch.tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
                     'document.match_score': torch.tensor([0, 0, 0, 0, 0, 1, 2, 3, 4, 5]),
                     'document.retrieval_score': torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}

    def test_select_docs_one_eg(self):
        for total, max_pos_docs in [(2, 1),
                                    (5, 2),
                                    (10, 5)]:

            output = SelectDocsOneEg(total=total, max_pos_docs=max_pos_docs)(deepcopy(self.data))

            for k, v in output.items():
                self.assertEqual(len(v), total)

            self.assertEqual(sum(output['document.match_score'] > 0), max_pos_docs)

    def test_always_same_length(self):

        for total, max_pos_docs in [(2, 1),
                                    (10, 1),
                                    (10, 3)]:

            output = SelectDocsOneEg(total=total,
                                     strict=False,
                                     max_pos_docs=max_pos_docs)(deepcopy(self.data))

            for k, v in output.items():
                self.assertEqual(len(v), total)

    def test_ordering(self):

        total, max_pos_docs = (10, 5)
        output = SelectDocsOneEg(total=total,
                                 strict=False,
                                 max_pos_docs=max_pos_docs)(deepcopy(self.data))

        self.assertEqual([x for x in output['document.feature']], [5, 4, 3, 2, 1, 10, 9, 8, 7, 6])
