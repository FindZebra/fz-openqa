from collections import defaultdict

import rich
import torch

from fz_openqa.modeling.heads import ClsHead
from fz_openqa.modeling.modules import ReaderMultipleChoice
from tests.modules.base import TestModel


class TestReaderMultipleChoice(TestModel):
    def setUp(self) -> None:
        super(TestReaderMultipleChoice, self).setUp()
        head = ClsHead(bert=self.bert, output_size=None)
        self.model = ReaderMultipleChoice(bert=self.bert, tokenizer=self.tokenizer, head=head)
        self.model.eval()

    def test_step(self):
        """Test that logits match the targets"""
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

    def test__expand_and_flatten_qd(self):
        output = self.model._expand_and_flatten_qd(self.batch, n_docs=self.n_documents)

        for key in ["document", "question"]:
            # check that "input_ids
            self.assertEqual(output[f"{key}.input_ids"].shape,
                             output[f"{key}.attention_mask"].shape)

            # check that all attributes are of shape [bs*n_docs, *]
            expected_shape = (
                self.batch_size * self.n_documents, self.batch[f"{key}.input_ids"].shape[-1])
            self.assertEqual(expected_shape, output[f"{key}.input_ids"].shape)

    def test__concat_questions_and_answers(self):
        """Check there is not padding between before the last SEP token."""
        for ordering in [["question", "document"], ["document", "question"]]:
            output = self.model._concat_questions_and_answers(self.batch, fields=ordering)

            tokens = output['input_ids']
            self.check_if_padding_last(tokens)
            self.check_if_only_one_cls_token(tokens)

    def check_if_only_one_cls_token(self, tokens: torch.Tensor):
        """Check there is one and only one CLS token in the sequence."""
        cls_id = self.tokenizer.cls_token_id
        for k in range(tokens.shape[0]):
            self.assertEqual(1, sum(tokens[k] == cls_id))

    def check_if_padding_last(self, tokens: torch.Tensor):
        pad_id = self.tokenizer.pad_token_id
        for k in range(tokens.shape[0]):
            padding_indexes = [i for i, t in enumerate(tokens[k]) if t == pad_id]
            if len(padding_indexes) > 0:
                first_pad_idx = min(padding_indexes)
                last_tokens = tokens[k][first_pad_idx:]
                if len(last_tokens) > 0:
                    self.assertTrue(all(int(pad_id) == int(t) for t in last_tokens))
