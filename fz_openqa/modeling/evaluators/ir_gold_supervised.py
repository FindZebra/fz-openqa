from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from fz_openqa.modeling.evaluators.abstract import Evaluator
from fz_openqa.modeling.similarities import Similarity
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import batch_reduce


class InformationRetrievalGoldSupervised(Evaluator):
    """
    Evaluates the Reader model `p(a_i | q, e, A)` using maximum likelihood estimation
    in a multiple choice QA context (A = [a_1,...a_P]). The loss is defined as:

        L =  sum_p log p(a_p | q, e, A) 1(p = a)

    where a is the index of the true answer.
    """

    _required_eval_feature_names = [
        "question.input_ids",
        "question.attention_mask",
        "question.input_ids",
        "document.attention_mask",
        "question.input_ids",
        "question.attention_mask",
        "rank",
    ]

    def __init__(self, similarity: Similarity, **kwargs):
        super().__init__(**kwargs)
        self.similarity = similarity

    def forward(
        self, model: nn.Module, batch: Batch, split: str, **kwargs: Any
    ) -> Batch:
        self.check_batch_type(batch)
        self.check_feature_names(batch)

        hq = model(
            batch=batch,
            model_key="question",  # todo: use question key (too much overfitting currently)
        )  # [bs, h]
        he = model(
            batch=batch,
            model_key="document",
        )  # [bs, h]

        logits = self.similarity(hq, he)  # [bs x bs]
        targets = (
            torch.arange(start=0, end=len(logits)).long().to(logits.device)
        )
        loss = F.cross_entropy(logits, targets, reduction="none")
        loss = batch_reduce(
            loss, torch.mean
        )  # keep one loss term per batch element

        return {
            "loss": loss,
            "preds": logits.argmax(dim=-1).detach(),
            "targets": targets,
        }
