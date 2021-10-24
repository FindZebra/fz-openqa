from copy import copy
from typing import List
from unittest import TestCase

import torch
from transformers import AutoTokenizer

from fz_openqa.datamodules.pipes.utils.collate_fn import (
    collate_and_pad_attributes, extract_and_collate_attributes_as_list)
from fz_openqa.utils.datastruct import Batch


def tensorize(examples: List[Batch]) -> List[Batch]:
    return [{k: torch.tensor(v) for k, v in d.items()} for d in examples]


class TestCollate(TestCase):
    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        # generate documents
        self.documents = [
            "Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.",
            "It also contains supporting code for evaluation and parameter tuning.",
            "Faiss is written in C++ with complete wrappers for Python (versions 2 and 3).",
        ]
        # encode and convert into examples
        encodings = self.tokenizer(self.documents)
        self.d_examples = [
            {f"document.{key}": encodings[key][idx] for key in encodings.keys()}
            for idx in range(len(self.documents))
        ]
        # add a non-sequential attribute
        for idx, d in enumerate(self.d_examples):
            d["document.idx"] = idx
        # tensorize
        self.d_examples = tensorize(self.d_examples)
        # add text attributes
        for ex, txt in zip(self.d_examples, self.documents):
            ex["document.text"] = txt

        # generate questions
        self.questions = [
            "Given a set of vectors x_i in dimension d",
            "Faiss builds a data structure in RAM.",
            "After the structure is constructed.",
        ]
        # encode and convert into examples
        encodings = self.tokenizer(self.questions)
        self.q_examples = [
            {f"question.{key}": encodings[key][idx] for key in encodings.keys()}
            for idx in range(len(self.questions))
        ]
        # tensorize
        self.q_examples = tensorize(self.q_examples)
        # add text attributes
        for ex, txt in zip(self.q_examples, self.questions):
            ex["question.text"] = txt

    def test_collate_and_pad_attributes(self):
        # encode the documents and questions
        d_batch = collate_and_pad_attributes(
            self.d_examples, tokenizer=self.tokenizer, key="document", exclude="text"
        )
        q_batch = collate_and_pad_attributes(
            self.q_examples,
            tokenizer=self.tokenizer,
            key="question",
            exclude=["token_type_ids", "text"],
        )

        # test the document batch
        self.assertTrue(all(isinstance(v, torch.Tensor) for v in d_batch.values()))
        self.assertTrue(all(v.shape[0] == len(self.d_examples) for v in d_batch.values()))
        self.assertTrue(all("document." in k for k in d_batch.keys()))
        for key in ["idx", "input_ids", "attention_mask", "token_type_ids"]:
            self.assertTrue(f"document.{key}")

        # test the question batch
        self.assertTrue(all(isinstance(v, torch.Tensor) for v in q_batch.values()))
        self.assertTrue(all(v.shape[0] == len(self.q_examples) for v in q_batch.values()))
        self.assertTrue(all("question." in k for k in q_batch.keys()))
        self.assertTrue(not all("token_type_ids" in k for k in q_batch.keys()))
        for key in ["idx", "input_ids", "attention_mask"]:
            self.assertTrue(f"question.{key}")

    def test_extract_and_collate_attributes_as_list(self):
        d_examples, d_batch_text = extract_and_collate_attributes_as_list(
            copy(self.d_examples), attribute="text"
        )
        self.assertTrue("document.text" in d_batch_text.keys())
        self.assertTrue(all(all(".text" not in k for k in d.keys()) for d in d_examples))
        self.assertTrue(len(d_batch_text["document.text"]) == len(d_examples))
