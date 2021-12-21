import torch

from fz_openqa.modeling.heads import ClsHead
from fz_openqa.modeling.modules import RetrieverSupervised
from tests.modules.base import TestModel


class TestRetrieverSupervised(TestModel):
    """Testing RetrieverSupervised. These tests rely on dummy data."""

    def setUp(self) -> None:
        super(TestRetrieverSupervised, self).setUp()
        head = ClsHead(bert=self.bert, output_size=None)
        self.model = RetrieverSupervised(bert=self.bert, tokenizer=self.tokenizer, head=head)
        self.model.eval()

    def test_forward(self):
        output = self.model.forward(self.batch)

        # score
        self.assertEqual([self.batch_size, self.batch_size * self.n_documents],
                         list(output['score'].shape))
        self.assertEqual([0, self.n_documents],
                         [x.detach().item() for x in output['score'].argmax(-1)])

        # logits
        self.assertIn("_hq_", output)
        self.assertIn("_hd_", output)
        # shape
        self.assertEqual(output["_hq_"].shape[:1], (self.batch_size,))
        self.assertEqual(output["_hd_"].shape[:1], (self.batch_size * self.n_documents,))

    def test__generate_targets(self):
        expected_targets = [0, 4]
        targets = [int(x) for x in
                   self.model._generate_targets(2, n_docs=4, device=torch.device('cpu'))]
        self.assertEqual(targets, expected_targets)

    def test_evaluate(self):
        """Call evaluate() and check that the logits match the generated targets."""
        output = self.model.evaluate(self.batch)
        # check logits
        self.assertEqual([x.item() for x in output['_logits_'].argmax(-1)],
                         [x.item() for x in output['_targets_']])

        # check that an exception is raised if `document.match_score` is absent
        batch = self.batch
        batch.pop('document.match_score')
        self.assertRaises(AssertionError, lambda: self.model.evaluate(batch))
