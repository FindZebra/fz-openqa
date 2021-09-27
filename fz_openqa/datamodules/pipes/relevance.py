from itertools import zip_longest
from typing import Any

import rich
import torch

from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.utils.datastruct import Batch


class RelevanceClassifier(Pipe):
    def __init__(
        self,
        question_field: str = "question.text",
        document_field: str = "document.text",
        output_key: str = "document.is_positive",
        output_count_key: str = "document.positive_count",
    ):
        self.output_key = output_key
        self.question_field = question_field
        self.document_field = document_field
        self.output_count_key = output_count_key

    def classify(self, question: Any, document: Any) -> bool:
        raise NotImplementedError

    def __call__(self, batch: Batch, **kwargs) -> Batch:
        results = [
            [self.classify(q, d) for d in docs]
            for q, docs in zip_longest(
                batch[self.question_field], batch[self.document_field]
            )
        ]
        results = torch.tensor(results)
        batch[self.output_key] = results
        batch[self.output_count_key] = results.float().sum(-1).long()
        return batch


class ExactMatch(RelevanceClassifier):
    def classify(self, question: Any, document: Any) -> bool:
        return bool(question in document)
