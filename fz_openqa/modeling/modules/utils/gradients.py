import abc
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import rich
import torch
from torch import Tensor
from torch.nn import functional as F

from .utils import batch_cartesian_product
from fz_openqa.utils.pretty import pprint_batch


class GradExpression(Enum):
    VARIATIONAL = "variational"
    IN_BATCH = "in_batch"
    REINFORCE = "reinforce"


@dataclass
class Quantities:
    """A small helper class to store the different terms involved in evaluating
    the likelihood."""

    score: Tensor
    logp_a__d: Tensor
    log_p_d__a: Tensor
    log_p_d__a_no_perm: Tensor
    logp_a: Tensor
    logp_a_star: Tensor
    logp_a_star__d: Tensor


class Estimator:
    def __init__(self, eval_topk: Optional[int] = None, debug: bool = False):
        self.eval_topk = eval_topk
        self.debug = debug

    @staticmethod
    def evaluate_likelihood_terms(
        retriever_score: Tensor,
        reader_score: Tensor,
        targets: torch.Tensor,
        eval_topk: Optional[int] = None,
        debug: bool = False,
    ) -> Quantities:
        # if `eval_topk` is provided, only keep the `eval_topk` documents with the
        # highest `retriever_score``
        if eval_topk is not None and retriever_score.shape[-1] > eval_topk:
            warnings.warn(f"Truncating scores {retriever_score.shape} to topk={eval_topk}")
            idx = torch.argsort(retriever_score, dim=-1, descending=True)[..., :eval_topk]
            retriever_score = retriever_score.gather(index=idx, dim=-1)
            reader_score = reader_score.gather(index=idx, dim=-1)

        targets = targets.unsqueeze(1)

        # repeat the scores for all combinations of documents:
        # `D \in D_1 \times D_2 \times ... \times D_N`
        expanded_reader_score, expanded_retriever_score = batch_cartesian_product(
            [reader_score, retriever_score]
        )
        # partial answer log-likelihood `\log p(a | q, D[\sigma], A)` for `\sigma \in S(M)`
        logp_a__d = expanded_reader_score.log_softmax(dim=1)
        targets_ = targets.expand(-1, expanded_reader_score.shape[2])
        logp_a_star__d = -F.cross_entropy(logp_a__d, targets_, reduction="none")

        # document log-likelihood `\log p(\sigma(d_j) | a_j, q_j)`
        normalizer = retriever_score.logsumexp(dim=2, keepdim=True)
        log_p_d__a = expanded_retriever_score - normalizer
        log_p_d__a_no_perm = retriever_score.log_softmax(dim=2)

        pprint_batch(
            {
                "retriever_score": retriever_score,
                "reader_score": reader_score,
                "expanded_reader_score": expanded_reader_score,
                "expanded_retriever_score": expanded_retriever_score,
                "log_p_d__a": log_p_d__a,
                "log_p_d__a_no_perm": log_p_d__a_no_perm,
            },
            "base_quantities_2",
            silent=not debug,
        )

        # answer log-likelihood: `\log p(a_\star | q, A)` (in-batch approximation)
        logp_a = (logp_a__d + log_p_d__a.sum(dim=1, keepdim=True)).logsumexp(dim=2)
        logp_a_star = torch.gather(logp_a, dim=1, index=targets).squeeze(1)
        return Quantities(
            score=reader_score,
            logp_a__d=logp_a__d,
            log_p_d__a=log_p_d__a,
            log_p_d__a_no_perm=log_p_d__a_no_perm,
            logp_a=logp_a,
            logp_a_star=logp_a_star,
            logp_a_star__d=logp_a_star__d,
        )

    def __call__(
        self,
        *,
        retriever_score: Tensor,
        reader_score: Tensor,
        targets: Tensor,
    ):
        q = self.evaluate_likelihood_terms(
            retriever_score, reader_score, targets, eval_topk=self.eval_topk, debug=self.debug
        )
        loss = self.compute_loss(q, targets=targets)
        return {
            "loss": loss,
            "reader/logp": q.logp_a_star.detach(),
            "_reader_logits_": q.logp_a.detach(),
            "_reader_targets_": targets.detach(),
            "_doc_logits_": q.log_p_d__a_no_perm.detach(),
        }

    @abc.abstractmethod
    def compute_loss(self, q: Quantities, *, targets: Tensor, **kwargs) -> torch.Tensor:
        ...


class InBatchGradients(Estimator):
    """Compute the gradients of the option retriever assuming the current
    batch to be the whole dataset."""

    def compute_loss(self, q: Quantities, *, targets: Tensor, **kwargs) -> torch.Tensor:
        return -1 * (q.logp_a_star).mean(-1)


class VariationalGradients(Estimator):
    """Compute the gradients using a Variational Lower Bound."""

    def compute_loss(self, q: Quantities, *, targets: Tensor, **kwargs) -> torch.Tensor:
        lb_logp_a = (q.logp_a__d + q.log_p_d__a.sum(dim=1, keepdim=True)).sum(dim=2)
        lb_logp_a_star = torch.gather(lb_logp_a, dim=1, index=targets[:, None]).squeeze(1)
        return -1 * (lb_logp_a_star).mean(-1)


class ReinforceGradients(Estimator):
    """Compute the gradients using Reinforce and assuming the proposal distributiuon to be the
    current retriever."""

    def compute_loss(self, q: Quantities, *, targets: Tensor, **kwargs) -> torch.Tensor:
        # todo: expand the `row_ids` and use masking row_ids[row_ids!=row_id] = 0
        # loss for the reader
        logp_a__d = q.logp_a__d.sum(2)
        logp_a_star__d = torch.gather(logp_a__d, dim=1, index=targets[:, None])
        reader_loss = -1 * logp_a_star__d

        # loss for the retriever
        logp_a__d_ = q.logp_a__d.detach()
        logp_a_hat = logp_a__d_.sum(dim=2, keepdim=True)
        logp_a_hat = 1.0 / logp_a__d_.size(2) * (logp_a_hat - logp_a__d_)
        Z = logp_a__d_ - logp_a_hat
        retriever_loss = -1 * (Z * q.log_p_d__a).sum(dim=(1, 2))

        return reader_loss + retriever_loss


def supervised_loss(partial_score: Tensor, match_score: Tensor, **kwargs):
    """
    Compute the supervised retrieval loss
    # todo: check loss, can we keep it without using the mask
    # todo: figure out how to compute the targets and logits for the metrics
    """

    pos_docs = match_score > 0
    loss_mask = pos_docs.sum(-1) > 0
    logits = partial_score[loss_mask]
    pos_docs = pos_docs[loss_mask].float()

    if logits.numel() > 0:
        n_total = len(pos_docs)
        n_pos = pos_docs.sum()
        loss = -(pos_docs * F.log_softmax(logits, dim=-1) / pos_docs.sum(dim=-1, keepdims=True))
        loss = loss.sum(-1)

    else:
        n_total = n_pos = 0
        loss = torch.tensor(0.0, dtype=partial_score.dtype, device=partial_score.device)

    # compute logits and targets for the metrics
    match_score = match_score[loss_mask]
    ids = torch.argsort(match_score, dim=-1, descending=True)
    targets = torch.zeros((logits.shape[0],), dtype=torch.long, device=logits.device)
    logits = logits.gather(index=ids, dim=-1)

    return {
        "retriever/loss": loss,
        "_retriever_logits_": logits,
        "_retriever_targets_": targets,
        "retriever/n_options": n_total,
        "retriever/n_positive": n_pos,
    }
