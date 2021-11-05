import os
from abc import ABC
from unittest import TestCase

import rich
import torch
from datasets import Dataset
from transformers import AutoTokenizer

from fz_openqa.datamodules.index import Index
from fz_openqa.datamodules.pipelines.collate import CollateTokens
from fz_openqa.datamodules.pipes import AddPrefix


class TestIndex(TestCase, ABC):
    _bert_id = "google/bert_uncased_L-2_H-128_A-2"
    cls: Index.__class__ = Index
    k = 3

    def setUp(self) -> None:
        torch.set_grad_enabled(False)
        os.environ['TOKENIZERS_PARALLELISM'] = "false"
        # init a tokenizer and bert
        self.tokenizer = AutoTokenizer.from_pretrained(self._bert_id)

        # Define dummy questions
        questions = ["Paris, France", "Banana, fruit"]
        encodings = self.tokenizer(questions, return_token_type_ids=False)
        questions = self.tokenizer.pad(encodings, return_tensors='pt')
        questions = AddPrefix(f"question.")(questions)
        self.dataset = Dataset.from_dict(questions)
        self.dataset.set_format("torch", output_all_columns=True)
        self.dataset_collate = self.collate = CollateTokens(tokenizer=self.tokenizer, prefix='question.')

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
        indexes = list(range(len(documents)))
        documents = self.tokenizer(documents, return_token_type_ids=False)
        documents = self.tokenizer.pad(documents, return_tensors='pt')
        documents = {**documents, 'row_idx': indexes}
        documents = AddPrefix(f"document.")(documents)
        self.corpus = Dataset.from_dict(documents)
        self.corpus.set_format("torch", output_all_columns=True)
        self.corpus_collate = self.collate = CollateTokens(tokenizer=self.tokenizer,
                                                            prefix='document.')


    def init_index(self) -> Index:
        return self.cls(dataset=self.corpus, k=self.k)
    # @staticmethod
    # def get_index_fingerprint(*, cls: Index.__class__, corpus: Dataset):
    #     return Pipe._fingerprint(cls(dataset=corpus))
    #
    # @abstractmethod
    # def test_fingerprint(self):
    #     pool = mp.Pool()
    #     fn = partial(TestIndex.get_index_fingerprint, cls=self.cls, corpus=self.corpus)
    #     fingerprints = pool.map(fn, range(3))
    #     rich.print(fingerprints)
    #
    # @abstractmethod
    # def test_build(self):
    #     assert False
    #
    # @abstractmethod
    # def test_search(self):
    #     assert False
    #
    # @abstractmethod
    # def test_get_example(self):
    #     assert False
