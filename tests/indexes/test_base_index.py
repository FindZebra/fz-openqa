import os
from abc import ABC
from unittest import TestCase

import datasets
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from fz_openqa.datamodules.index import Index
from fz_openqa.datamodules.index.search_result import SearchResult
from fz_openqa.datamodules.pipelines.collate import CollateTokens
from fz_openqa.datamodules.pipes import AddPrefix, Parallel, Collate
from fz_openqa.utils.train_utils import setup_safe_env, silent_huggingface


def cast_to_array(x):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


class TestIndex(TestCase, ABC):
    # todo: test search_one
    _bert_id = "google/bert_uncased_L-2_H-128_A-2"
    cls: Index.__class__ = Index
    k = 100
    pt_cols = ["input_ids", "attention_mask"]

    def setUp(self) -> None:
        """setup a toy dataset where retrieval outcomes are expected."""

        torch.set_grad_enabled(False)
        silent_huggingface()
        datasets.set_caching_enabled(False)
        setup_safe_env()
        os.environ['TOKENIZERS_PARALLELISM'] = "false"
        # init a tokenizer and bert
        self.tokenizer = AutoTokenizer.from_pretrained(self._bert_id)

        # Define dummy questions
        questions_text = ["Paris, France", "Banana, fruit"]
        encodings = self.tokenizer(questions_text, return_token_type_ids=False)
        questions = self.tokenizer.pad(encodings, return_tensors='pt')
        questions = {**questions, 'text': questions_text}
        questions = AddPrefix(f"question.")(questions)

        # convert to HF dataset
        self.dataset = Dataset.from_dict(questions)
        self.dataset.set_format("torch", output_all_columns=True,
                                columns=[f"question.{attr}" for attr in self.pt_cols])
        self.dataset_collate = Parallel(Collate(['question.text']),
                                        CollateTokens(tokenizer=self.tokenizer, prefix='question.'))

        # Define documents, including one for each question [#0, #4]
        documents_text = [
            "Paris in France",
            "Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.",
            "It also contains supporting code for evaluation and parameter tuning.",
            "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3).",
            "Banana is a fruit",
            "Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.",
            "It also contains supporting code for evaluation and parameter tuning.",
            "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3).",
        ]
        indexes = list(range(len(documents_text)))
        documents = self.tokenizer(documents_text, return_token_type_ids=False)
        documents = self.tokenizer.pad(documents, return_tensors='pt')
        documents = {**documents, 'row_idx': indexes, 'text': documents_text}
        documents = AddPrefix(f"document.")(documents)

        # convert to HF dataset
        self.corpus = Dataset.from_dict(documents)
        self.corpus.set_format("torch", output_all_columns=True,
                               columns=[f"document.{attr}" for attr in self.pt_cols])
        self.corpus_collate = Parallel(Collate(['document.text', 'document.row_idx']),
                                       CollateTokens(tokenizer=self.tokenizer, prefix='document.'))

        # define targets
        self.retrieval_targets = [0, 4]

    def _init_index(self) -> Index:
        """Init the index."""
        return self.cls(dataset=self.corpus, k=self.k)

    def _test_dill_inspect(self):
        """Check if the Index can be pickled."""
        index = self._init_index()
        dill_status = index.dill_inspect()
        if isinstance(dill_status, bool):
            self.assertTrue(dill_status, f"Index {type(index).__name__} could not be pickled.")
        elif isinstance(dill_status, dict):
            for k, v in dill_status.items():
                self.assertTrue(v, f"Attribute {k} could not be pickled.")
        else:
            raise ValueError(f"dill_inspect() returned unexpected type {type(dill_status)}.")

    def _test_is_indexed(self):
        """Check that `is_index` attribute."""
        index = self._init_index()
        self.assertTrue(index.is_indexed)

    def _test_search(self):
        """
        Check the outcome of the search results:
        1. output must be of the right type and shape
        2. scores must be sorted
        3. the closest match must corresponds to the dummy data
        """
        index = self._init_index()
        # build the query and search the index using the query
        query = self.dataset_collate([row for row in self.dataset])
        output = index.search(query, k=self.k)
        # check the output type
        self.assertIsInstance(output, SearchResult)

        # cast output
        data = {'score': output.score, 'index': output.index}
        data = {k: cast_to_array(v) for k, v in data.items()}

        # check that the index values are in [0, len(self.dataset) - 1]
        self.assertTrue((data['index'] >= 0).all())
        self.assertTrue((data['index'] < len(self.corpus)).all())

        expected_shape = (len(self.dataset), self.k)
        for key, d in data.items():
            self.assertEqual(d.shape, expected_shape)

        # check the top-1 scores
        for target, scores, idx in zip(self.retrieval_targets, data['score'], data['index']):
            pred = np.argmax(scores, axis=-1)
            self.assertEqual(target, idx[pred], "top retrieved document is not the expected one")
            # check if output values are sorted
            self.assertTrue(np.all(scores[:-1] >= scores[1:]), "scores are not properly sorted")
