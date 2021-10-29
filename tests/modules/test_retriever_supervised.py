from copy import deepcopy
from unittest import TestCase

import torch
from transformers import AutoTokenizer, AutoModel, BertPreTrainedModel

from fz_openqa.datamodules.pipes import AddPrefix, ApplyToAll
from fz_openqa.modeling.backbone import BertLinearHeadCls
from fz_openqa.modeling.modules import RetrieverSupervised


class TestRetrieverSupervised(TestCase):
    """Testing RetrieverSupervised. These tests rely on dummy data."""

    def setUp(self) -> None:
        self.batch_size = 2
        self.n_documents = 4

        # init a tokenizer and bert
        self.tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        bert: BertPreTrainedModel = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        backbone = BertLinearHeadCls(bert=bert, output_size=None)
        self.model = RetrieverSupervised(bert=bert, tokenizer=self.tokenizer, backbone=backbone)
        self.model.eval()

        # Define dummy questions
        questions = ["Paris, France", "Banana, fruit"]
        encodings = self.tokenizer(questions, return_token_type_ids=False)
        q_batch = self.tokenizer.pad(encodings, return_tensors='pt')
        q_batch = AddPrefix(f"question.")(q_batch)

        # Define documents, including one for each question [#0, #4]
        documents = [
            "Paris in France",
            "Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.",
            "It also contains supporting code for evaluation and parameter tuning.",
            "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3).",
            "Banana is a fruit",
            "Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.",
            "It also contains supporting code for evaluation and parameter tuning.",
            "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3).",
        ]
        encodings = self.tokenizer(documents, return_token_type_ids=False)
        d_batch = self.tokenizer.pad(encodings, return_tensors='pt')
        d_batch = AddPrefix(f"document.")(d_batch)
        d_batch['document.match_score'] = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0])
        d_batch = ApplyToAll(lambda x: x.view(2, 4, *x.shape[1:]))(d_batch)

        # store all
        self._batch = {**q_batch, **d_batch}
        assert self._batch['document.input_ids'].shape[:2] == (
            self.batch_size, self.n_documents), "batch is not properly initialized"

    @property
    def batch(self):
        return deepcopy(self._batch)

    @torch.no_grad()
    def test_step(self):
        scores = self.model.step(self.batch)
        # keys
        self.assertIn("_hq_", scores)
        self.assertIn("_hd_", scores)
        # dimension
        self.assertEqual(scores["_hq_"].dim(), 2)
        self.assertEqual(scores["_hd_"].dim(), 2)
        # shape
        self.assertEqual(scores["_hq_"].shape[:1], (self.batch_size,))
        self.assertEqual(scores["_hd_"].shape[:1], (self.batch_size * self.n_documents,))

    def test_forward(self):
        output = self.model.forward(self.batch)
        self.assertEqual(output['score'].shape,
                         [self.batch_size, self.batch_size * self.n_documents])
        self.assertEqual([x.detach().item() for x in output['score'].argmax(-1)],
                         [0, self.batch_size])

    def test__reduce_step_output(self):
        data = {"_hd_": torch.randn((self.batch_size * self.n_documents, 8)),
                "_hq_": torch.randn((self.batch_size, 8))}

        # call the model
        output = self.model._reduce_step_output(data)

        # check keys
        for key in ["_logits_", "_targets_", "loss", "n_options"]:
            self.assertIn(key, output)

        self.assertEqual(output["_logits_"].shape,
                         (self.batch_size, self.batch_size * self.n_documents))
        self.assertEqual(output["n_options"], self.batch_size * self.n_documents)
        self.assertEqual(output["_targets_"].shape, (self.batch_size,))

    def test__generate_targets(self):
        expected_targets = [0, 4]
        targets = [int(x) for x in
                   self.model._generate_targets(2, n_docs=4, device=torch.device('cpu'))]
        self.assertEqual(targets, expected_targets)

    @torch.no_grad()
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
