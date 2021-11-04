from abc import abstractmethod, ABC
from copy import deepcopy
from unittest import TestCase

import torch
from transformers import AutoTokenizer, BertPreTrainedModel, AutoModel

from fz_openqa.datamodules.pipes import AddPrefix, ApplyToAll


class TestModel(TestCase, ABC):
    """Testing RetrieverSupervised. These tests rely on dummy data."""
    _bert_id = "google/bert_uncased_L-2_H-128_A-2"

    @abstractmethod
    def setUp(self) -> None:
        """Instantiate the TestCase with dummy data"""


        self.batch_size = 2
        self.n_documents = 4
        self.n_options = 3

        # init a tokenizer and bert
        self.tokenizer = AutoTokenizer.from_pretrained(self._bert_id)
        self.bert: BertPreTrainedModel = AutoModel.from_pretrained(self._bert_id)

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
        d_batch = ApplyToAll(lambda x: x.view(self.batch_size, self.n_documents, *x.shape[1:]))(
            d_batch)

        # define the answers
        answers = ["Paris France", "Rocket", "Bike", "Truck", "Banana fruit", "Aircraft"]
        encodings = self.tokenizer(answers, return_token_type_ids=False)
        a_batch = self.tokenizer.pad(encodings, return_tensors='pt')
        a_batch = AddPrefix(f"answer.")(a_batch)
        a_batch = ApplyToAll(lambda x: x.view(self.batch_size, self.n_options, *x.shape[1:]))(
            a_batch)
        # answers are "Paris France" and "Banana fruit"
        a_batch['answer.target'] = torch.tensor([0, 1])

        # store all
        self._batch = {**q_batch, **d_batch, **a_batch}
        assert self._batch['document.input_ids'].shape[:2] == (
            self.batch_size, self.n_documents), "batch is not properly initialized"

    @property
    def batch(self):
        return deepcopy(self._batch)
