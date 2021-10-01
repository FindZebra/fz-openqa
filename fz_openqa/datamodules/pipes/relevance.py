from typing import Any
from typing import Dict

import torch

from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch


class RelevanceClassifier(Pipe):
    def __init__(
        self,
        answer_prefix: str = "answer.",
        document_prefix: str = "document.",
        output_key: str = "document.is_positive",
    ):
        self.output_key = output_key
        self.answer_prefix = answer_prefix
        self.document_prefix = document_prefix

    def classify(self, question: Any, document: Any) -> bool:
        raise NotImplementedError

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        results = []
        batch_size = len(next(iter(batch.values())))
        for i in range(batch_size):
            a_data_i = {
                k: v[i] for k, v in batch.items() if self.answer_prefix in k
            }
            d_data_i = {
                k: v[i] for k, v in batch.items() if self.document_prefix in k
            }

            # iterate through each document
            results_i = []
            n_docs = len(next(iter(d_data_i.values())))
            for j in range(n_docs):
                d_data_ij = {k: v[j] for k, v in d_data_i.items()}
                results_i += [self.classify(a_data_i, d_data_ij)]
            results += [results_i]

        results = torch.tensor(results)
        batch[self.output_key] = results
        return batch


class ExactMatch(RelevanceClassifier):
    def classify(
        self, answer: Dict[str, Any], document: Dict[str, Any]
    ) -> bool:
        doc_text = document["document.text"]
        answer_index = answer["answer.target"]
        answer_text = answer["answer.text"][answer_index]
        return bool(answer_text in doc_text)
