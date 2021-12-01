import warnings
from dataclasses import dataclass
from enum import Enum

import rich
import torch
from torch import Tensor
from torch.nn import functional as F

from .utils import batch_cartesian_product
from .utils import check_only_first_doc_positive


class GradExpression(Enum):
    BATCH_SUM = "batch-sum"
    BATCH_GATHER = "batch-gather"
    VARIATIONAL = "variational"


@dataclass
class Quantities:
    score: Tensor
    log_p_d__a: Tensor
    log_p_d__a_no_perm: Tensor
    logp_a: Tensor
    logp_a_star: Tensor
    logp_a_star__d: Tensor


def base_quantities(partial_score, targets) -> Quantities:
    # repeat the scores for all combinations of documents: `\sigma \in S(M)`
    score = batch_cartesian_product(partial_score)
    # partial answer log-likelihood `\log p(a | q, D[\sigma], A)` for `\sigma \in S(M)`
    logp_a__d = score.log_softmax(dim=1)
    targets_ = targets.unsqueeze(1).expand(-1, score.shape[2])
    logp_a_star__d = -F.cross_entropy(logp_a__d, targets_, reduction="none")
    # document log-likelihood `\log p(\sigma(d_j) | a_j, q_j)`
    normalizer = partial_score.logsumexp(dim=2, keepdim=True)
    log_p_d__a = score - normalizer
    log_p_d__a_no_perm = partial_score.log_softmax(dim=2)
    # answer lower-bound: `\sum_\sigma \log p(a_\star | q, A)`
    logp_a = (logp_a__d * log_p_d__a.exp()).sum(-1)
    logp_a_star = torch.gather(logp_a, dim=1, index=targets[:, None]).squeeze(1)
    return Quantities(
        score=score,
        log_p_d__a=log_p_d__a,
        log_p_d__a_no_perm=log_p_d__a_no_perm,
        logp_a=logp_a,
        logp_a_star=logp_a_star,
        logp_a_star__d=logp_a_star__d,
    )


def in_batch_grads(
    partial_score: Tensor, targets: Tensor, *, grad_expr: GradExpression = GradExpression.BATCH_SUM
):
    """Compute the gradients assuming the batch being the entire dataset"""
    q = base_quantities(partial_score, targets)

    # gradients / loss
    loss_reader = q.logp_a_star__d
    retriever_score = q.logp_a_star__d.unsqueeze(1) * q.log_p_d__a.sum(1, keepdim=True).exp()
    part_loss_retriever = retriever_score.detach() * q.log_p_d__a
    if grad_expr == GradExpression.BATCH_SUM:
        loss_retriever = part_loss_retriever.sum(1)
    elif grad_expr == GradExpression.BATCH_GATHER:
        targets_ = targets[:, None, None].expand(-1, 1, part_loss_retriever.shape[2])
        loss_retriever = part_loss_retriever.gather(dim=1, index=targets_).squeeze(1)
    else:
        raise ValueError(
            f"Unknown GradExpression: {grad_expr}, " f"GradExpression={GradExpression}"
        )
    loss = -1 * (loss_reader + loss_retriever).mean(-1)
    return {
        "loss": loss,
        "reader/logp": q.logp_a_star.detach(),
        "_reader_logits_": q.logp_a.detach(),
        "_reader_targets_": targets.detach(),
        "_doc_logits_": q.log_p_d__a_no_perm.detach(),
    }


def supervised_loss(partial_score: Tensor, match_score: Tensor, **kwargs):
    """Compute the supervised retrieval loss"""
    if not torch.all(match_score[..., 1:] == 0):
        warnings.warn("Not all documents with index >0 are negative.")

    pos_docs = match_score > 0
    loss_mask = (pos_docs[:, :, 0]) & (pos_docs[:, :, 1:].sum(-1) == 0)
    logits = partial_score[loss_mask]
    targets = torch.zeros((logits.shape[0],), dtype=torch.long, device=logits.device)
    if logits.numel() > 0:
        loss = F.cross_entropy(logits, targets)
    else:
        loss = 0.0

    return {"retriever_loss": loss, "_retriever_logits_": logits, "_retriever_targets_": targets}


def variational_grads(partial_score: Tensor, targets: Tensor, **kwargs):
    """Compute the gradients using the variational approximation"""
    q = base_quantities(partial_score, targets)

    # retrieval loss term `\sum_j \sum_d ( \d f(a, d) / \d \theta - \sum_d \d f(a, d) / \d \theta)`
    retrieval_loss = (q.score - q.score.mean(-1, keepdim=True)).mean(dim=(1, 2))

    # loss
    loss_reader = q.logp_a_star__d.mean(1)
    loss = -1 * (loss_reader + retrieval_loss)

    return {
        "loss": loss,
        "reader/logp": q.logp_a_star.detach(),
        "_reader_logits_": q.logp_a.detach(),
        "_reader_targets_": targets.detach(),
        "_doc_logits_": q.log_p_d__a_no_perm.detach(),
    }
