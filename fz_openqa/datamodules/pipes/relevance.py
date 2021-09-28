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
        output_count_key: str = "document.positive_count",
    ):
        self.output_key = output_key
        self.answer_prefix = answer_prefix
        self.document_prefix = document_prefix
        self.output_count_key = output_count_key

    def classify(self, question: Any, document: Any) -> bool:
        raise NotImplementedError

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        results = []
        batch_size = len(next(iter(batch.values())))
        for i in range(batch_size):
            q_data_i = {
                k: v[i] for k, v in batch.items() if self.answer_prefix in k
            }
            d_data_i = {
                k: v[i] for k, v in batch.items() if self.document_prefix in k
            }

            # iterate through each document
            results_i = []
            n_docs = len(next(iter(d_data_i)))
            for j in range(n_docs):
                d_data_ij = {k: v[j] for k, v in d_data_i.items()}
                results_i += [self.classify(q_data_i, d_data_ij)]
            results += [results_i]

        results = torch.tensor(results)
        batch[self.output_key] = results
        batch[self.output_count_key] = results.float().sum(-1).long()
        return batch


class ExactMatch(RelevanceClassifier):
    def classify(
        self, answer: Dict[str, Any], document: Dict[str, Any]
    ) -> bool:
        doc_text = document["document.text"]
        answer_index = answer["answer.target"]
        answer_text = answer["answer.text"][answer_index]
        return bool(answer_text in doc_text)
