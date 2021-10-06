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
        batch_size = self.batch_size(batch)
        for i in range(batch_size):
            eg_ans_i = self.get_eg(
                batch,
                i,
                filter_op=lambda key: str(key).startswith(self.answer_prefix),
            )
            eg_doc_i = self.get_eg(
                batch,
                i,
                filter_op=lambda key: str(key).startswith(
                    self.document_prefix
                ),
            )

            # iterate through each document
            results_i = []
            n_docs = len(next(iter(eg_doc_i.values())))
            for j in range(n_docs):
                d_data_ij = {k: v[j] for k, v in eg_doc_i.items()}
                results_i += [self.classify(eg_ans_i, d_data_ij)]
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
        return bool(answer_text.lower() in doc_text.lower())
