from typing import Any
from typing import Dict

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric

from .abstract import Evaluator
from fz_openqa.utils.functional import batch_reduce


class MultipleChoiceQaMaximumLikelihood(Evaluator):
    """
    Evaluates the Reader model `p(a_i | q, e, A)` using maximum likelihood estimation
    in a multiple choice QA context (A = [a_1,...a_P]). The loss is defined as:

        L =  sum_p log p(a_p | q, e, A) 1(p = a)

    where a is the index of the true answer.
    """

    # TODO: formalize the Metrics logic (when to compute and log)
    text_key = "question"
    _required_eval_feature_names = [
        "answer_idx",
    ]

    def get_metric(self, split: str) -> Metric:
        return self.metrics[f"_{split}"]

    def forward(
        self, model: nn.Module, batch: Any, split: str, **kwargs: Any
    ) -> Dict[str, Tensor]:
        self.check_batch_type(batch)
        self.check_feature_names(batch)

        logits: Tensor = model(batch)
        targets: Tensor = batch["answer_idx"]
        loss = F.cross_entropy(logits, targets, reduction="none")
        loss = batch_reduce(
            loss, torch.mean
        )  # keep one loss term per batch element
        return {
            "loss": loss,
            "logits": logits.detach(),
            "targets": targets.detach(),
        }
