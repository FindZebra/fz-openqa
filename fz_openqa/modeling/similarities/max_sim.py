import torch
from torch import Tensor
from transformers import AutoModel

from .base import Similarity

bert_id = "google/bert_uncased_L-2_H-128_A-2"


class MaxSim(Similarity):
    def __init__(self, similarity_metric=str):
        assert similarity_metric in {"cosine", "l2"}
        self.similarity_metric = similarity_metric
        self.bert = AutoModel.from_pretrained(bert_id)
        self.bert.eval()

    def __call__(self, Q: Tensor, D: Tensor) -> Tensor:
        """Compute the maximum similarity score between a query and a document.

        Parameters
        ----------
        similarity_metric
            Specify similarity: cosine or L2
        query
            Query representation as Tensor
        document
            Document representation as Tensor

        Returns
        -------
        scores

        """
        document = self.doc_representations(D["input_ids"], D["attention_mask"])
        query = self.query_representations(Q["input_ids"], Q["attention_mask"])
        if self.similarity_metric == "cosine":
            scores = (query @ document.permute(0, 2, 1)).max(2).values.sum(1)
        elif self.similarity_metric == "l2":
            scores = (
                (-1.0 * ((query.unsqueeze(2) - document.unsqueeze(1)) ** 2).sum(-1))
                .max(-1)
                .values.sum(-1)
            )

        return scores

    def doc_representations(self, input_ids, attention_mask):
        """ Compute document representations using BERT """
        document = self.bert(input_ids, attention_mask=attention_mask)[0]

        return document

    def query_representations(self, input_ids, attention_mask):
        """ Compute query representations using BERT """
        query = self.bert(input_ids, attention_mask=attention_mask)[0]

        return query
