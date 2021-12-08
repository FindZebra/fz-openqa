import unittest
from copy import deepcopy
from functools import partial

import numpy as np
import rich
import torch
from torch import Tensor
from torch.optim import Adam

from fz_openqa.datamodules.pipelines.collate.field import CollateField
from fz_openqa.datamodules.pipelines.preprocessing import FormatAndTokenize
from fz_openqa.datamodules.pipes import TextFormatter, Parallel, Sequential, Apply, ConcatTextFields
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.datamodules.pipes.nesting import Expand, ApplyAsFlatten
from fz_openqa.datamodules.utils.transformations import add_spec_token
from fz_openqa.modeling.heads import ClsHead
from fz_openqa.modeling.modules import OptionRetriever
from fz_openqa.utils.pretty import pprint_batch
from tests.modules.base import TestModel


class TestOptionRetriever(TestModel):
    """Testing OptionRetriever. Tests rely on dummy data."""

    # Define dummy questions
    questions = ["Where is Paris?", "What is a Banana?"]

    # Define documents, including one for each question [#0, #4]
    documents = [
        [
            ["Paris is in France.",
             "Faiss is a library for efficient similarity search and clustering of dense vectors. "
             "It contains algorithms that search in sets of vectors of any size, "
             "up to ones that possibly do not fit in RAM.",
             "It also contains supporting code for evaluation and parameter tuning.",
             "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3)."],
            ["Rockets are use to fly to space",
             "Faiss is a library for efficient similarity search and clustering of dense vectors. "
             "It contains algorithms that search in sets of vectors of any size, "
             "up to ones that possibly do not fit in RAM.",
             "It also contains supporting code for evaluation and parameter tuning.",
             "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3)."],
            ["Bike is the vehicle of the future",
             "Faiss is a library for efficient similarity search and clustering of dense vectors. "
             "It contains algorithms that search in sets of vectors of any size, "
             "up to ones that possibly do not fit in RAM.",
             "It also contains supporting code for evaluation and parameter tuning.",
             "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3)."],
        ],
        [
            ["Truck can carry tons of merchandises.",
             "Faiss is a library for efficient similarity search and clustering of dense vectors. "
             "It contains algorithms that search in sets of vectors of any size, up to ones "
             "that possibly do not fit in RAM.",
             "It also contains supporting code for evaluation and parameter tuning.",
             "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3)."],
            ["A banana is a fruit",
             "Faiss is a library for efficient similarity search and clustering of dense vectors. "
             "It contains algorithms that search in sets of vectors of any size, up to ones "
             "that possibly do not fit in RAM.",
             "It also contains supporting code for evaluation and parameter tuning.",
             "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3)."],
            ["Aircraft can fly across oceans.",
             "Faiss is a library for efficient similarity search and clustering of dense vectors. "
             "It contains algorithms that search in sets of vectors of any size, up to ones "
             "that possibly do not fit in RAM.",
             "It also contains supporting code for evaluation and parameter tuning.",
             "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3)."]
        ]
    ]
    match_score = torch.tensor([3 * [[1, 0, 0, 0]], 3 * [[1, 0, 0, 0]]])

    # dummy answers
    answers = [["France", "Rocket", "Bike"], ["Truck", "Fruit", "Aircraft"]]
    answer_targets = torch.tensor([0, 1])

    def setUp(self) -> None:
        super(TestOptionRetriever, self).setUp()
        head = ClsHead(bert=self.bert, output_size=None)
        self.model = OptionRetriever(bert=self.bert, tokenizer=self.tokenizer, head=head)
        self.model.eval()

    def get_preprocessing_pipe(self):
        # concat question and answer
        add_spec_tokens_pipe = Apply(
            {"question.text": partial(add_spec_token, self.tokenizer.sep_token)},
            element_wise=True
        )
        concat_qa = Sequential(
            add_spec_tokens_pipe,
            Expand(axis=1, n=self.n_options, update=True, input_filter=In(["question.text"])),
            ApplyAsFlatten(
                ConcatTextFields(keys=["answer.text", "question.text"],
                                 new_key="question.text"),
                level=1,
            ),
            input_filter=In(["question.text", "answer.text"]),
            update=True
        )

        # format and tokenize args
        args = {'text_formatter': TextFormatter(),
                'tokenizer': self.tokenizer,
                'max_length': 512,
                'add_special_tokens': True,
                'spec_token': None,
                }
        preprocess = Parallel(
            FormatAndTokenize(
                prefix="question.",
                shape=[-1, self.n_options],
                **args
            ),
            FormatAndTokenize(
                prefix="document.",
                shape=[-1, self.n_options, self.n_documents],
                **args
            ),
            update=True
        )
        collate = Parallel(
            CollateField("document",
                         tokenizer=self.tokenizer,
                         level=2,
                         to_tensor=["match_score"]
                         ),
            CollateField("question",
                         tokenizer=self.tokenizer,
                         level=1,
                         ),
            CollateField("answer",
                         include_only=["target"],
                         to_tensor=["target"],
                         level=0,
                         ),
        )
        return Sequential(concat_qa, preprocess, collate)

    def test_step(self):
        """test the `step` method, make sure that the output has the right keys
        and check that _logits_ match the _targets_."""
        output = self.model.step(self.batch)
        # keys
        self.assertIn("loss", output)
        self.assertIn("_reader_logits_", output)
        self.assertIn("_reader_targets_", output)

        # check that model predictions are correct
        self.assertEqual(output['_reader_logits_'].argmax(-1).numpy().tolist(),
                         output['_reader_targets_'].numpy().tolist())

        # check that probs sum to 1 `logits.exp().sum(-1) == 1`
        self.assertTrue(torch.allclose(output['_reader_logits_'].exp().sum(-1), torch.tensor(1.)))

    def test_compute_score(self):
        """test the `_compute_score` method. Make sure that retrieval score
        match the targets `document.match_score`"""
        batch = self.batch
        output = self.model.forward(batch)
        hq, hd = (output[k] for k in ["_hq_", "_hd_"])
        score = self.model._compute_score(hq=hq, hd=hd)
        targets: Tensor = batch['document.match_score'].argmax(-1)
        preds: Tensor = score.argmax(-1)
        self.assertTrue((targets == preds).numpy().all())

    def test_forward(self):
        """Test the shape of the returned tensors in `forward`"""
        output = self.model.forward(self.batch)
        self.assertEqual([self.batch_size, self.n_options, self.n_documents],
                         list(output['_hd_'].shape[:3]))
        self.assertEqual([self.batch_size, self.n_options],
                         list(output['_hq_'].shape[:2]))


    @unittest.skip("not implemented")
    def test__reduce_step_output(self):
        data = {"loss": torch.tensor([0.4, 0.6]),
                "logp": torch.tensor([0.4, 0.6])}

        output = self.model._reduce_step_output(data)
        for key in output:
            self.assertEqual(output[key], 0.5)

    @torch.enable_grad()
    def test_overfit(self):
        """Add noise to the weights of the model and optimize for a few steps."""
        VERBOSE = True
        if VERBOSE:
            np.set_printoptions(precision=3, suppress=True)

        # gather data
        batch = self.batch
        doc_targets = batch['document.match_score']

        # take a copy of the model, and add noise to the weights
        model = deepcopy(self.model)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn(param.size()) * 0.3)

        # optimize the model
        model.train()
        optimizer = Adam(model.parameters(), lr=1e-4)
        n_steps = 100
        for i in range(n_steps):
            optimizer.zero_grad()
            output = model.evaluate(batch)
            loss = output['loss'].mean()
            loss.backward()
            optimizer.step()

            if VERBOSE and i % 10 == 0:
                probs = output['_reader_logits_'].detach().exp().numpy()
                rich.print(f"{i} - loss={loss:.3f}, probs[0]={probs[0]}, probs[1]={probs[1]}")
                doc_probs = output['_doc_logits_'].detach().exp()
                # doc_dist = (doc_probs - doc_targets).pow(2)
                # rich.print(f"-- doc_probs: \n{doc_probs.numpy()}")

        # check that the model puts high probs on the correct answer
        targets = batch['answer.target']
        output = model.evaluate(batch)
        probs = output['_reader_logits_'].detach().exp()
        target_probs = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1).numpy()
        probs = probs.numpy()
        targets = targets.numpy()
        if VERBOSE:
            rich.print(f">> targets: {targets}")
            rich.print(f">>> probs: {probs}")
            rich.print(f">>> probs.target: {target_probs}")

            doc_score = output['_doc_logits_'].detach().exp()
            rich.print(f"doc probs: \n{doc_score.numpy()}")

        self.assertTrue((target_probs > 0.9).all())
