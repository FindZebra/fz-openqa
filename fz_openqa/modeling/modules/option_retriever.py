from enum import Enum
from typing import Any
from typing import Optional

import torch
from datasets import Split
from torch import Tensor

from .utils.gradients import GradExpression
from .utils.gradients import in_batch_grads
from .utils.gradients import supervised_loss
from .utils.gradients import variational_grads
from .utils.utils import flatten_first_dims
from fz_openqa.modeling.modules.base import Module
from fz_openqa.utils.datastruct import Batch


class Similarity(Enum):
    CLS = "cls"
    COLBERT = "colbert"


class OptionRetriever(Module):
    """
    A model for multiple-choice OpenQA.
    This is a retriever-only model allowing both for retrieval and option selection.
    The model is described in : https://hackmd.io/tQ4_EDx5TMyQwwWO1rvUIA
    """

    _required_feature_names = []

    _required_eval_feature_names = [
        "question.input_ids",
        "question.attention_mask",
        "document.input_ids",
        "document.attention_mask",
        "answer.target",
    ]

    # prefix for the logged metrics
    task_id: Optional[str] = None

    # metrics to display in the progress bar
    pbar_metrics = [
        "train/reader/logp",
        "validation/reader/logp",
        "train/reader/Accuracy",
        "validation/reader/Accuracy",
    ]

    # require heads
    _required_heads = ["question", "document"]

    def __init__(
        self,
        *args,
        alpha: float = 0.01,
        grad_expr: GradExpression = GradExpression.VARIATIONAL,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.grad_expr = GradExpression(grad_expr)
        head = next(iter(self.heads.values()))
        self.similarity = Similarity(head.id)
        self.alpha = alpha

    def _init_metrics(self, prefix: str):
        """Initialize the metrics for each split."""
        self.metrics = self._get_base_metrics(prefix=f"{prefix}reader/")

        self.retriever_metrics = self._get_base_metrics(
            prefix=f"{prefix}retriever/", topk=[None, 3, 5, 10, 20, 50, 100]
        )

    def _forward(self, batch: Batch, **kwargs) -> Batch:
        q_shape = d_shape = None
        output = {}
        if "document.input_ids" in batch:
            d_shape = batch["document.input_ids"].shape
            # flatten the batch_size and n_options and n_docs dimensions
            d_batch = flatten_first_dims(
                batch,
                n_dims=3,
                keys=["document.input_ids", "document.attention_mask"],
            )

            # process the document with the backbone
            hd = self._backbone(d_batch, prefix="document", head="document", **kwargs)
            # reshape and return
            hd = hd.reshape(*d_shape[:3], *hd.shape[1:])
            output["_hd_"] = hd

        if "question.input_ids" in batch:
            q_shape = batch["question.input_ids"].shape
            # flatten the batch_size and n_options dimensions
            q_batch = flatten_first_dims(
                batch,
                n_dims=2,
                keys=["question.input_ids", "question.attention_mask"],
            )

            # process the document with the backbone
            hq = self._backbone(q_batch, prefix="question", head="question", **kwargs)

            # reshape and return
            hq = hq.reshape(*q_shape[:2], *hq.shape[1:])
            output["_hq_"] = hq

        if all(d is not None for d in (d_shape, q_shape)):
            if not d_shape[:2] == q_shape[:2]:
                raise ValueError(
                    f"Expected 2 first dimensions to be equal, "
                    f"got documents of shape: {d_shape} and "
                    f"questions of shape{q_shape}"
                )

        return output

    def _step(self, batch: Batch, **kwargs: Any) -> Batch:
        """
        Compute the forward pass for the question and the documents.
        """
        # check features, check that the first document of each question is positive
        # process the batch using BERT and the heads
        targets = batch["answer.target"]
        output = self._forward(batch, **kwargs)
        hq, hd = (output[k] for k in ["_hq_", "_hd_"])

        # compute the score for each pair `f([q_j; a_j], d_jk)`
        partial_score = self._compute_score(hd=hd, hq=hq)

        if self.grad_expr in [GradExpression.BATCH_GATHER, GradExpression.BATCH_SUM]:
            step_output = in_batch_grads(partial_score, targets, grad_expr=self.grad_expr)
        elif self.grad_expr == GradExpression.VARIATIONAL:
            step_output = variational_grads(partial_score, targets, grad_expr=self.grad_expr)
        else:
            raise ValueError(f"Unknown grad_expr: {self.grad_expr}")

        # auxiliary loss
        supervised_loss_out = supervised_loss(partial_score, batch["document.match_score"])
        supervised_loss_ = supervised_loss_out.get("retriever_loss", 0)
        if self.alpha > 0:
            step_output["loss"] += self.alpha * supervised_loss_
        step_output.update(supervised_loss_out)

        return step_output

    def _compute_score(self, *, hd: Tensor, hq: Tensor) -> Tensor:
        """compute the score for each option and document $f([q;a], d)$ : [bs, n_options, n_docs]"""
        if self.similarity == Similarity.CLS:
            return torch.einsum("boh, bodh -> bod", hq, hd)
        elif self.similarity == Similarity.COLBERT:
            scores = torch.einsum("bouh, bodvh -> boduv", hq, hd)
            max_scores, _ = scores.max(-1)
            return max_scores.sum(-1)
        else:
            raise ValueError(f"Unknown similarity: {self.similarity}, Similarity={Similarity}")

    def _reduce_step_output(self, output: Batch) -> Batch:
        """
        Gather losses and logits from all devides and return
        """

        # average losses
        for k in ["loss", "reader/logp"]:
            y = output.get(k, None)
            if y is not None:
                output[k] = y.mean()

        return output

    def update_metrics(self, output: Batch, split: Split) -> None:
        """update the metrics of the given split."""
        logits, targets = (output.get(k, None) for k in ("_reader_logits_", "_reader_targets_"))
        self.metrics.update(split, logits, targets)

        retrieval_logits, retrieval_targets = (
            output.get(k, None) for k in ("_retriever_logits_", "_retriever_targets_")
        )
        if retrieval_logits is not None and retrieval_logits.numel() > 0:
            self.retriever_metrics.update(split, retrieval_logits, retrieval_targets)

    def reset_metrics(self, split: Optional[Split] = None) -> None:
        """
        Reset the metrics corresponding to `split` if provided, else
        reset all the metrics.
        """
        self.metrics.reset(split)
        self.retriever_metrics.reset(split)

    def compute_metrics(self, split: Optional[Split] = None) -> Batch:
        """
        Compute the metrics for the given `split` else compute the metrics for all splits.
        The metrics are return after computation.
        """
        return {
            **self.metrics.compute(split),
            **self.retriever_metrics.compute(split),
        }
