from enum import Enum
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .utils import batch_cartesian_product
from .utils import flatten_first_dims
from fz_openqa.modeling.modules.base import Module
from fz_openqa.utils.datastruct import Batch


class GradExpression(Enum):
    SUM = "sum"
    GATHER = "gather"


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
    task_id: Optional[str] = "reader"

    # metrics to display in the progress bar
    pbar_metrics = [
        "train/reader/logp",
        "validation/reader/logp",
        "train/reader/Accuracy",
        "validation/reader/Accuracy",
    ]

    # require heads
    _required_heads = ["question", "document"]

    def __init__(self, *args, grad_expr: GradExpression = GradExpression.SUM, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_expr = GradExpression(grad_expr)

    def _init_metrics(self, prefix: str):
        """Initialize the metrics for each split."""
        self.metrics = self._get_base_metrics(prefix=f"{prefix}")

    def _forward(self, batch: Batch, **kwargs) -> Batch:

        d_shape = batch["document.input_ids"].shape
        q_shape = batch["question.input_ids"].shape
        if not d_shape[:2] == q_shape[:2]:
            raise ValueError(
                f"Expected 2 first dimensions to be equal, "
                f"got documents of shape: {d_shape} and "
                f"questions of shape{q_shape}"
            )

        # flatten the batch_size and n_options and n_docs dimensions
        d_batch = flatten_first_dims(
            batch,
            n_dims=3,
            keys=["document.input_ids", "document.attention_mask"],
        )

        # flatten the batch_size and n_options dimensions
        q_batch = flatten_first_dims(
            batch,
            n_dims=2,
            keys=["question.input_ids", "question.attention_mask"],
        )

        # process the document with the backbones
        hd = self._backbone(d_batch, prefix="document", head="document", **kwargs)
        hq = self._backbone(q_batch, prefix="question", head="question", **kwargs)

        # reshape and return
        hd = hd.reshape(*d_shape[:3], *hd.shape[1:])
        hq = hq.reshape(*q_shape[:2], *hq.shape[1:])
        return {"_hd_": hd, "_hq_": hq}

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

        # repeat the scores for all combinations of documents: `\sigma \in S(M)`
        score = batch_cartesian_product(partial_score)

        # partial answer log-likelihood `\log p(a | q, D[\sigma], A)` for `\sigma \in S(M)`
        logp_a__d = score.log_softmax(dim=1)
        targets_ = targets.unsqueeze(1).expand(-1, score.shape[2])
        # rich.print(f">> logp_a__d={logp_a__d.shape}, targets_={targets_.shape}")
        logp_a_star__d = -F.cross_entropy(logp_a__d, targets_, reduction="none")

        # document log-likelihood `\log p(\sigma(d_j) | a_j, q_j)`
        normalizer = partial_score.logsumexp(dim=2, keepdim=True)
        log_p_d__a = score - normalizer
        log_p_d__a_no_perm = partial_score.log_softmax(dim=2)

        # answer lower-bound: `\sum_\sigma \log p(a_\star | q, A)`
        logp_a = (logp_a__d * log_p_d__a.exp()).sum(-1)
        logp_a_star = torch.gather(logp_a, dim=1, index=targets[:, None]).squeeze(1)

        # gradients / loss
        loss_reader = logp_a_star__d
        retriever_score = logp_a_star__d.unsqueeze(1) * log_p_d__a.sum(1, keepdim=True).exp()
        part_loss_retriever = retriever_score.detach() * log_p_d__a
        if self.grad_expr == GradExpression.SUM:
            loss_retriever = part_loss_retriever.sum(1)
        elif self.grad_expr == GradExpression.GATHER:
            targets_ = targets[:, None, None].expand(-1, 1, part_loss_retriever.shape[2])
            loss_retriever = part_loss_retriever.gather(dim=1, index=targets_).squeeze(1)
        else:
            raise ValueError(
                f"Unknown GradExpression: {self.grad_expr}, " f"GradExpression={GradExpression}"
            )
        loss = -1 * (loss_reader + loss_retriever).mean(-1)

        return {
            "loss": loss,
            "logp": logp_a_star.detach(),
            "_logits_": logp_a.detach(),
            "_targets_": targets.detach(),
            "_doc_logits_": log_p_d__a_no_perm.detach(),
        }

    def _compute_score(self, *, hd: Tensor, hq: Tensor) -> Tensor:
        """compute the score for each option and document $f([q;a], d)$ : [bs, n_options, n_docs]"""
        return torch.einsum("boh, bodh -> bod", hq, hd)

    def _reduce_step_output(self, output: Batch) -> Batch:
        """
        Gather losses and logits from all devides and return
        """

        # average losses
        for k in ["loss", "logp"]:
            y = output.get(k, None)
            if y is not None:
                output[k] = y.mean()

        return output
