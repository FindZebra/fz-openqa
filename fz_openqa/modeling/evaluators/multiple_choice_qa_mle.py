from copy import copy
from typing import Any
from typing import Dict
from typing import Optional

import torch
from datasets import Split
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy

from .base import BaseEvaluator
from .metrics import SplitMetrics
from .utils import check_first_doc_positive
from .utils import expand_and_flatten
from .utils import flatten_first_dims
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import batch_reduce


class MultipleChoiceQaMaximumLikelihood(BaseEvaluator):
    """
    Evaluates the Reader model `p(a_i | q, d_1,...d_m, A)` using maximum likelihood estimation
    in a multiple choice QA context (A = [a_1,...a_P]) and using a supervised selection model.
    The answering loss loss is defined as:

        L =  sum_{p, i} log p(a_p | q, d_i, A) 1(p = a)

    where a is the index of the true answer.

    The selection model (eq. 5 in DPR) corresponds to retrieving the positive document among
    the pos+neg documents for a given question.
    """

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

    def init_metrics(self, prefix: str = ""):
        """Initialize a Metric for each split=train/validation/test
        fir both the answering model and the selection model"""
        metric_kwargs = {"compute_on_step": False, "dist_sync_on_step": True}

        def init_answer_metric():
            return MetricCollection([Accuracy(**metric_kwargs)], prefix=prefix)

        self.answer_metrics = SplitMetrics(init_answer_metric)

        def init_selection_metric():
            return MetricCollection(
                [Accuracy(**metric_kwargs)], prefix=f"{prefix}selection-"
            )

        self.selection_metrics = SplitMetrics(init_selection_metric)

    def forward(
        self, model: nn.Module, batch: Batch, split: str, **kwargs: Any
    ) -> Dict[str, Tensor]:
        self.check_batch_type(batch)
        self.check_feature_names(batch)

        # arguments
        batch = copy(batch)  # make sure the original batch is not modified
        bs, n_docs = batch["document.input_ids"].shape[:2]

        # check that the first document of each question is positive
        check_first_doc_positive(batch)

        # flatten documents of shape [bs, n_docs, T] to [bs*n_docs, T]
        batch.update(
            **flatten_first_dims(
                batch,
                2,
                keys=["document.input_ids", "document.attention_mask"],
            )
        )

        # expand questions to shape [bs, n_docs, L] and flatten to shape [bs*n_docs, L]
        batch.update(
            **expand_and_flatten(
                batch,
                n_docs,
                keys=["question.input_ids", "question.attention_mask"],
            )
        )

        # forward pass through the reader model
        answer_logits, select_logits = model(batch)

        # reader loss
        answer_targets: Tensor = batch["answer_idx"]
        answer_loss = self.compute_answer_loss(
            answer_logits, answer_targets, bs, n_docs
        )

        # selection loss
        # It is assumed that there is a single positive document and its index is zero
        select_targets = torch.zeros_like(batch["answer_idx"]).long()
        select_loss, select_targets = self.compute_selection_loss(
            select_targets, select_logits
        )

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
            "answer_logits": answer_logits.detach(),
            "answer_targets": answer_targets.detach(),
            "select_logits": select_logits.detach(),
            "select_targets": select_targets.detach(),
        }

    def compute_selection_loss(self, select_targets, select_logits):
        """
        L =  - log p(idx_pos | docs[pos+neg])
        """
        select_loss = F.cross_entropy(
            select_logits, select_targets, reduction="none"
        )
        select_loss = batch_reduce(
            select_loss, torch.mean
        )  # keep one loss term per batch element
        return select_loss, select_targets

    def compute_answer_loss(self, answer_logits, answer_targets, bs, n_docs):
        """L = 1/N_docs \\sum_i - log p(a | q, d_i, A)"""
        answer_loss = F.cross_entropy(
            answer_logits.permute(0, 2, 1),
            answer_targets[:, None].expand(bs, n_docs),
            reduction="none",
        ).mean(1)
        answer_loss = batch_reduce(
            answer_loss, torch.mean
        )  # keep one loss term per batch element
        return answer_loss

    def forward_end(self, output: Batch, split: Split) -> Any:
        """Apply a post-processing step to the forward method.
        The output is the output of the forward method.

        This method is called after the `output` has been gathered
        from each device. This method must aggregate the loss across
        devices.

        torchmetrics update() calls should be placed here.
        The output must at least contains the `loss` key.
        """

        for k in ["loss", "select_loss", "answer_loss"]:
            y = output.get(k, None)
            if y is not None:
                output[k] = y.mean()

        self.update_metrics(output, split)

        for k in [
            "answer_logits",
            "answer_targets",
            "select_logits",
            "select_targets",
        ]:
            output.pop(k, None)
        return output

    def update_metrics(self, output: Batch, split: Split) -> None:
        """update the metrics of the given split."""
        answer_logits, answer_targets = (
            output.get(k, None) for k in ("answer_logits", "answer_targets")
        )
        self.answer_metrics.update(split, answer_logits, answer_targets)

        select_logits, select_targets = (
            output.get(k, None) for k in ("select_logits", "select_targets")
        )
        if select_targets is not None:
            self.answer_metrics.update(split, answer_logits, answer_targets)

    def reset_metrics(self, split: Optional[Split] = None) -> None:
        """
        Reset the metrics corresponding to `split` if provided, else
        reset all the metrics.
        """
        self.answer_metrics.reset(split)
        self.selection_metrics.reset(split)

    def compute_metrics(self, split: Optional[Split] = None) -> Batch:
        """
        Compute the metrics for the given `split` else compute the metrics for all splits.
        The metrics are return after computation.
        """
        return {
            **self.answer_metrics.compute(split),
            **self.selection_metrics.compute(split),
        }
