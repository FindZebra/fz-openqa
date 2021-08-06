from copy import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
from datasets import Split
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy

from .base import BaseEvaluator
from .metrics import SplitMetrics
from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.functional import batch_reduce


class NestedMetricCollections(MetricCollection):
    """
    A class that allows handling multiple sub-MetricCollections, each of them index by a key.
    Only the signature of the update method changes, which requires a dictionary of tuples as input.
    """

    def __init__(self, metrics: Dict[str, MetricCollection]):
        nn.Module.__init__(self)
        self.metrics = nn.ModuleDict(metrics)

    def update(self, values=Dict[str, Tuple[Tensor]]) -> None:
        for k, v in values.items():
            self.metrics[k].update(*v)

    def compute(self) -> Any:
        return {
            k: v
            for metric in self.metrics.values()
            if next(iter(metric.values())).mode is not None
            for k, v in metric.compute().items()
        }

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()


class MultipleChoiceQaMaximumLikelihood(BaseEvaluator):
    """
    Evaluates the Reader model `p(a_i | q, d_1,...d_m, A)` using maximum likelihood estimation
    in a multiple choice QA context (A = [a_1,...a_P]). The loss is defined as:

        L =  sum_{p, i} log p(a_p | q, d_i, A) 1(p = a)

    where a is the index of the true answer.
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
        """Initialize a Metric for each split=train/validation/test"""
        metric_kwargs = {"compute_on_step": False, "dist_sync_on_step": True}

        def gen_answer_metric():
            return MetricCollection([Accuracy(**metric_kwargs)], prefix=prefix)

        self.answer_metrics = SplitMetrics(gen_answer_metric)

        def gen_selection_metric():
            return MetricCollection(
                [Accuracy(**metric_kwargs)], prefix=f"{prefix}selection-"
            )

        self.selection_metrics = SplitMetrics(gen_selection_metric)

    def forward(
        self, model: nn.Module, batch: Batch, split: str, **kwargs: Any
    ) -> Dict[str, Tensor]:
        self.check_batch_type(batch)
        self.check_feature_names(batch)

        # arguments
        batch = copy(batch)  # make sure the original batch is not modified
        bs, n_docs = batch["document.input_ids"].shape[:2]

        # check that the first document of each question is positive
        assert torch.all(batch["document.is_positive"][:, 0] == 1)
        if batch["document.is_positive"].shape[1] > 1:
            assert torch.all(batch["document.is_positive"][:, 1:] == 0)

        # flatten documents
        batch.update(
            **self.flatten_documents(
                batch,
                2,
                keys=["document.input_ids", "document.attention_mask"],
            )
        )

        # expand questions and flatten
        batch.update(
            **self.expand_and_flatten(
                batch,
                n_docs,
                keys=["question.input_ids", "question.attention_mask"],
            )
        )

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
            "answer_logits": answer_logits.detach(),
            "answer_targets": answer_targets.detach(),
            "select_logits": select_logits.detach(),
            "select_targets": select_targets.detach(),
        }

    @staticmethod
    def flatten_documents(batch: Batch, n_dims, *, keys: List[str]) -> Batch:
        output = {}
        for k in keys:
            output[k] = batch[k].view(-1, *batch[k].shape[n_dims:])
        return output

    @staticmethod
    def expand_and_flatten(batch: Batch, n_docs, *, keys: List[str]) -> Batch:
        output = {}
        for k in keys:
            v = batch[k]
            v = (
                v[:, None]
                .expand(v.shape[0], n_docs, *v.shape[1:])
                .contiguous()
            )
            output[k] = v.view(-1, *v.shape[2:])
        return output

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
