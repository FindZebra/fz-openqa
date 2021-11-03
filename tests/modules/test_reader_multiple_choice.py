from collections import defaultdict

import torch

from fz_openqa.modeling.heads import ClsHead
from fz_openqa.modeling.modules import ReaderMultipleChoice
from fz_openqa.utils.pretty import pprint_batch
from tests.modules.base import TestModel


class TestReaderMultipleChoice(TestModel):
    def setUp(self) -> None:
        super(TestReaderMultipleChoice, self).setUp()
        heads = defaultdict(lambda: ClsHead(bert=self.bert, output_size=None))
        self.model = ReaderMultipleChoice(bert=self.bert, tokenizer=self.tokenizer, heads=heads)
        self.model.eval()

    @torch.no_grad()
    def test_step(self):
        """Test that logits match targets"""
        output = self.model.step(self.batch)

        # keys
        self.assertIn("loss", output)
        self.assertIn("_relevance_targets_", output)
        self.assertIn("_answer_logits_", output)
        self.assertIn("_relevance_logits_", output)
        # shape
        self.assertEqual(output["_answer_logits_"].shape,
                         (self.batch_size, self.n_options))
        self.assertEqual(output["_relevance_logits_"].shape,
                         (self.batch_size, self.n_documents,))

    @torch.no_grad()
    def test_forward(self):
        """Test the shape of tensors return by `forward`"""
        output = self.model.forward(self.batch)

        # keys
        self.assertIn("_answer_logits_", output)
        self.assertIn("_relevance_logits_", output)
        # shape
        self.assertEqual(output["_answer_logits_"].shape,
                         (self.batch_size, self.n_documents, self.n_options))
        self.assertEqual(output["_relevance_logits_"].shape,
                         (self.batch_size, self.n_documents,))

    def test__reduce_step_output(self):
        data = {"loss": torch.tensor([0.4, 0.6]),
                "relevance_loss": torch.tensor([0.4, 0.6]),
                "answer_loss": torch.tensor([0.4, 0.6])}

        output = self.model._reduce_step_output(data)
        for key in output:
            self.assertEqual(output[key], 0.5)

    def test__concat_fields_across_dim_one(self):
        # output = self.model._concat_fields_across_dim_one(self.batch, fields=["question", "answer"])
        # todo
        # pprint_batch(output,  "_concat_fields_across_dim_one")
        pass

    def test__expand_and_flatten_qd(self):
        pass

    def test_expand_and_flatten(self):
        pass
