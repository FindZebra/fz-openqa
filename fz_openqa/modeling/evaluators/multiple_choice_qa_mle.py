from typing import Any
from typing import Dict

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric

from .abstract import Evaluator
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import pprint_batch
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
        "question.input_ids",
        "question.attention_mask",
        "document.input_ids",
        "document.attention_mask",
        "document.is_positive",
        "answer_choices.input_ids",
        "answer_choices.attention_mask",
    ]

    def get_metric(self, split: str) -> Metric:
        return self.metrics[f"_{split}"]

    def forward(
        self, model: nn.Module, batch: Batch, split: str, **kwargs: Any
    ) -> Dict[str, Tensor]:
        self.check_batch_type(batch)
        self.check_feature_names(batch)
        bs, n_docs = batch["document.input_ids"].shape[:2]

        # check that the first document of each question is positive
        assert torch.all(batch["document.is_positive"][:, 0] == 1)
        if batch["document.is_positive"].shape[1] > 1:
            assert torch.all(batch["document.is_positive"][:, 1:] == 0)

        # forward pass through the reader model
        answer_logits, select_logits = model(batch)

        # reader loss
        answer_targets: Tensor = batch["answer_idx"]
        answer_loss = F.cross_entropy(
            answer_logits.permute(0, 2, 1),
            answer_targets[:, None].expand(bs, n_docs),
            reduction="none",
        ).mean(1)

        answer_loss = batch_reduce(
            answer_loss, torch.mean
        )  # keep one loss term per batch element

        # selection loss: here it is assumed that there is a single positive document
        # and its index is zero
        select_targets = torch.zeros_like(batch["answer_idx"]).long()
        select_loss = F.cross_entropy(
            select_logits, select_targets, reduction="none"
        )
        select_loss = batch_reduce(
            select_loss, torch.mean
        )  # keep one loss term per batch element

        # final loss
        loss = answer_loss + select_loss

        # select ``answer_logits`` corresponding to the argmax of th ``select_logits``
        # to return for the computation of the accuracy
        selected = select_logits.argmax(-1)
        _index = selected.view(bs, 1, 1).expand(bs, 1, answer_logits.shape[-1])
        answer_logits = answer_logits.gather(dim=1, index=_index).squeeze(1)

        return {
            "loss": loss,
            "select_loss": select_loss.detach(),
            "answer_loss": answer_loss.detach(),
            "logits": answer_logits.detach(),
            "targets": answer_targets.detach(),
        }

    def forward_end(self, output: Batch, split: str) -> Any:
        """Apply a post-processing step to the forward method.
        The output is the output of the forward method.

        This method is called after the `output` has been gathered
        from each device. This method must aggregate the loss across
        devices.

        torchmetrics update() calls should be placed here.
        The output must at least contains the `loss` key.
        """

        output["loss"] = output["loss"].mean()
        output["select_loss"] = output["select_loss"].mean()
        output["answer_loss"] = output["answer_loss"].mean()
        self.update_metrics(output, split)
        output.pop("preds", None)
        output.pop("logits", None)
        output.pop("targets", None)
        return output
