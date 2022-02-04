import unittest
from copy import deepcopy, copy
from functools import partial

import numpy as np
import rich
import torch
from pytorch_lightning import seed_everything
from torch import Tensor
from torch.optim import Adam

from fz_openqa.datamodules.pipelines.collate.field import CollateField
from fz_openqa.datamodules.pipelines.preprocessing import FormatAndTokenize
from fz_openqa.datamodules.pipes import TextFormatter, Parallel, Sequential, Apply, \
    ConcatTextFields, PrioritySampler
from fz_openqa.datamodules.pipes.control.condition import In, HasPrefix
from fz_openqa.datamodules.pipes.nesting import Expand, ApplyAsFlatten, Nested
from fz_openqa.datamodules.utils.transformations import add_spec_token
from fz_openqa.modeling.heads import ColbertHead, ClsHead
from fz_openqa.modeling.modules import OptionRetriever
from fz_openqa.modeling.modules.utils.gradients import GradExpression, InBatchGradients
from fz_openqa.utils.pretty import get_separator
from tests.modules.base import TestModel


class TestOptionRetriever(TestModel):
    """Testing OptionRetriever. Tests rely on dummy data."""
    # Define dummy questions
    # _bert_id = "nboost/pt-bert-base-uncased-msmarco"
    questions = ["What color is shrubbery?", "What size is a banana?"]

    # Define documents, including one for each question [#0, #4]
    documents = [
        [
            ["a shrubbery is purple.",
             "Faiss is a library for efficient similarity search and clustering of dense vectors. ",
             "It also contains supporting code for evaluation and parameter tuning.",
             "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3).",
             "random gibberish",
             "hello world"
             ],
            [
                "Faiss is a library for efficient similarity search and clustering of dense vectors.",
                "a shrubbery is not red.",
                "It also contains supporting code for evaluation and parameter tuning.",
                "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3).",
                "random gibberish",
                "hello world"
            ],
            [
                "Faiss is a library for efficient similarity search and clustering of dense vectors.",
                "It also contains supporting code for evaluation and parameter tuning.",
                "a shrubbery is not blue.",
                "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3).",
                "random gibberish",
                "hello world"
            ],
        ],
        [
            [
                "Faiss is a library for efficient similarity search and clustering of dense vectors. "
                "It contains algorithms that search in sets of vectors of any size, up to ones "
                "that possibly do not fit in RAM.",
                "It also contains supporting code for evaluation and parameter tuning.",
                "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3).",
                "a banana is not small sized",
                "random gibberish",
                "hello world"
            ],
            [
                "Faiss is a library for efficient similarity search and clustering of dense vectors. ",
                "It also contains supporting code for evaluation and parameter tuning.",
                "A banana is medium sized.",
                "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3).",
                "random gibberish",
                "hello world"
            ],
            [
                "Faiss is a library for efficient similarity search and clustering of dense vectors. ",
                "A banana is not large sized.",
                "It also contains supporting code for evaluation and parameter tuning.",
                "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3).",
                "random gibberish",
                "hello world"
            ]
        ]
    ]
    match_score = torch.tensor([[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
                                [[0, 0, 0, 1, 0 ,0], [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0]]])
    doc_ids = torch.arange(0, match_score.numel(), 1).view(match_score.shape)

    # dummy answers
    answers = [["purple", "red", "blue"],
               ["small", "medium", "large"]]
    answer_targets = torch.tensor([0, 1])

    def setUp(self) -> None:
        super(TestOptionRetriever, self).setUp()
        head = ClsHead(bert=self.bert, output_size=None)
        self.model = OptionRetriever(bert=self.bert,
                                     tokenizer=self.tokenizer,
                                     head=head,
                                     grad_expr=GradExpression("reinforce"))
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
                'spec_tokens': None,
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
                         to_tensor=["match_score", "row_idx"]
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

    @unittest.skip("TODO")
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

    @unittest.skip("TODO")
    def test_compute_score(self):
        """test the `_compute_score` method. Make sure that retrieval score
        match the targets `document.match_score`"""
        batch = self.batch
        output = self.model.forward(batch)
        hq, hd = (output[k] for k in ["_hq_retriever_", "_hd_retriever_"])
        score = self.model._compute_score(hq=hq, hd=hd)
        targets: Tensor = batch['document.match_score'].argmax(-1)
        preds: Tensor = score.argmax(-1)
        self.assertTrue((targets == preds).numpy().all())

    @unittest.skip("TODO")
    def test_forward(self):
        """Test the shape of the returned tensors in `forward`"""
        output = self.model.forward(self.batch)
        self.assertEqual([self.batch_size, self.n_options, self.n_documents],
                         list(output['_hd_retriever_'].shape[:3]))
        self.assertEqual([self.batch_size, self.n_options],
                         list(output['_hq_retriever_'].shape[:2]))
        self.assertEqual([self.batch_size, self.n_options, self.n_documents],
                         list(output['_hd_reader_'].shape[:3]))
        self.assertEqual([self.batch_size, self.n_options],
                         list(output['_hq_reader_'].shape[:2]))

    @torch.enable_grad()
    def test_overfit(self):
        """Add noise to the weights of the model and optimize for a few steps."""
        VERBOSE = 1
        # seed_everything(1)
        if VERBOSE:
            np.set_printoptions(precision=3, suppress=True)

        # gather data
        ref_batch = self.batch
        doc_targets = ref_batch['document.match_score']
        sampler = Nested(PrioritySampler(total=3, update=True), level=2,
                         input_filter=HasPrefix("document."), update=True)

        # take a copy of the model, and add noise to the weights
        model = deepcopy(self.model)
        model.eval()
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn(param.size()) * 0)


        # compute the retrieval_score using `InBatchGradients
        with torch.no_grad():
            _estimator = model.estimator
            model.estimator = InBatchGradients()
            output = model.evaluate(ref_batch)
            model.estimator = _estimator
            doc_logits = output['_doc_logits_'].detach()
            # doc_logits.zero_()
            doc_logits /= 100
            # doc_logits = doc_logits + torch.randn_like(doc_logits)
            ref_batch['document.retrieval_score'] = doc_logits
            rich.print(f"retrieval_probs: {doc_logits.softmax(dim=-1).numpy()}")

        # optimize the model
        # model.train()
        optimizer = Adam(model.parameters(), lr=1e-5)
        n_steps = 100
        for i in range(n_steps):
            batch = sampler(deepcopy(ref_batch))
            optimizer.zero_grad()
            output = model.evaluate(batch)
            loss = output['loss'].mean()
            loss.backward()
            optimizer.step()

            if VERBOSE and i % 10 == 0:
                probs = output['_reader_logits_'].exp().detach().numpy()
                doc_probs = output['_doc_logits_'].detach()
                # doc_dist = (doc_probs - doc_targets).pow(2)
                rich.print(f"{i} - loss={loss:.3f},"
                           f" probs[0]={probs[0]}, "
                           f"probs[1]={probs[1]}")


                if VERBOSE > 5:
                    rich.print(f"-- doc_probs: \n{doc_probs.numpy()}")

        # check that the model puts high probs on the correct answer
        batch = deepcopy(ref_batch)
        log_pz = doc_logits.zero_()
        log_pz.log_softmax(dim=-1)
        retrieval_log_Z = log_pz.logsumexp(dim=-1, keepdim=True).expand_as(log_pz)
        batch[f"document.retrieval_log_prob"] = log_pz
        batch[f"document.retrieval_log_Z"] = retrieval_log_Z
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

            doc_score = output['_doc_logits_'].softmax(-1).detach()
            print(get_separator())
            rich.print(f"doc logits: \n{doc_score.numpy()}")
            print(get_separator())
            rich.print(f"doc targets: \n{doc_targets.numpy()}")

        self.assertTrue((probs.argmax(-1) == targets).all())
