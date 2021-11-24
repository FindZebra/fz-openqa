from abc import ABC
from copy import deepcopy
from unittest import TestCase

import torch
from transformers import AutoTokenizer, BertPreTrainedModel, AutoModel

from fz_openqa.datamodules.pipelines.collate.field import CollateField
from fz_openqa.datamodules.pipelines.preprocessing import FormatAndTokenize
from fz_openqa.datamodules.pipes import TextFormatter, Parallel, Sequential
from fz_openqa.utils.datastruct import Batch


class TestModel(TestCase, ABC):
    """Testing RetrieverSupervised. These tests rely on dummy data."""
    _bert_id = "google/bert_uncased_L-2_H-128_A-2"

    # Define dummy questions
    questions = ["Paris, France", "Banana, fruit"]

    # Define documents, including one for each question [#0, #4]
    documents = [
        ["Paris in France",
         "Faiss is a library for efficient similarity search and clustering of dense vectors. "
         "It contains algorithms that search in sets of vectors of any size, "
         "up to ones that possibly do not fit in RAM.",
         "It also contains supporting code for evaluation and parameter tuning.",
         "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3)."],
        ["Banana is a fruit",
         "Faiss is a library for efficient similarity search and clustering of dense vectors. "
         "It contains algorithms that search in sets of vectors of any size, up to ones "
         "that possibly do not fit in RAM.",
         "It also contains supporting code for evaluation and parameter tuning.",
         "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3)."]
    ]
    match_score = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])

    # dummy answers
    answers = [["Paris France", "Rocket", "Bike"], ["Truck", "Banana fruit", "Aircraft"]]
    answer_targets = torch.tensor([0, 1])

    def setUp(self) -> None:
        """Instantiate the TestCase with dummy data"""
        torch.set_grad_enabled(False)
        self.batch_size = 2
        self.n_documents = 4
        self.n_options = 3

        # init a tokenizer and bert
        self.tokenizer = AutoTokenizer.from_pretrained(self._bert_id)
        self.bert: BertPreTrainedModel = AutoModel.from_pretrained(self._bert_id)

        # tokenize data
        self._batch = self._encode_data(self.data)

        assert self._batch['document.input_ids'].shape[:2] == (
            self.batch_size, self.n_documents), "batch is not properly initialized"

    def _encode_data(self, data: Batch) -> Batch:
        pipe = self.get_preprocessing_pipe()
        return pipe(data)

    def get_preprocessing_pipe(self):
        args = {'text_formatter': TextFormatter(),
                'tokenizer': self.tokenizer,
                'max_length': 512,
                'add_special_tokens': True,
                'spec_token': None,
                }
        preprocess = Parallel(
            FormatAndTokenize(
                prefix="question.",
                shape=[-1],
                **args
            ),
            FormatAndTokenize(
                prefix="document.",
                shape=[-1, self.n_documents],
                **args
            ),
            FormatAndTokenize(
                prefix="answer.",
                shape=[-1, self.n_options],
                **args
            ),
            update=True
        )
        collate = Parallel(
            CollateField("document",
                         tokenizer=self.tokenizer,
                         level=1,
                         to_tensor=["match_score"]
                         ),
            CollateField("question",
                         tokenizer=self.tokenizer,
                         level=0,
                         ),
            CollateField("answer",
                         tokenizer=self.tokenizer,
                         exclude=["target"],
                         level=1,
                         ),
            CollateField("answer",
                         include_only=["target"],
                         to_tensor=["target"],
                         level=0,
                         ),
        )
        pipe = Sequential(preprocess, collate)
        return pipe

    @property
    def batch(self):
        return deepcopy(self._batch)

    @property
    def data(self):
        return {
            "question.text": self.questions,
            "document.text": self.documents,
            "answer.text": self.answers,
            "answer.target": self.answer_targets,
            "document.match_score": self.match_score
        }
