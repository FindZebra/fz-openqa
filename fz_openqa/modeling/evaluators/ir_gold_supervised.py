from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from fz_openqa.modeling.evaluators.abstract import Evaluator
from fz_openqa.modeling.similarities import Similarity
from fz_openqa.utils.datastruct import Batch


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

        return {
            "hq": hq,
            "he": he,
        }

    def post_forward(self, output: Batch, split: str) -> Any:
        """Apply a post-processing step to the forward method.
        The output is the output of the forward method.

        This method is called after the `output` has been gathered
        from each device. This method must aggregate the loss across
        devices.

        torchmetrics update() calls should be placed here.
        The output must at least contains the `loss` key.
        """
        hq, he = (output.pop(k) for k in ["hq", "he"])
        logits = self.similarity(hq, he)  # [bs x bs]
        targets = (
            torch.arange(start=0, end=len(logits)).long().to(logits.device)
        )
        loss = F.cross_entropy(logits, targets, reduction="mean")
        output["loss"] = loss.mean()
        output["batch_size"] = logits.shape[1]  # store the batch size
        output["preds"] = logits.argmax(-1)
        output["targets"] = targets
        self.update_metrics(output, split)
        output.pop("preds")
        output.pop("targets")
        return output
