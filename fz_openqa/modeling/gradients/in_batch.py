import warnings
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from fz_openqa.modeling.gradients.base import Gradients
from fz_openqa.modeling.gradients.retriever_diagnostics import retriever_diagnostics
from fz_openqa.modeling.gradients.utils import batch_cartesian_product
from fz_openqa.modeling.gradients.utils import kl_divergence
from fz_openqa.utils.functional import batch_reduce
from fz_openqa.utils.pretty import pprint_batch


@dataclass
class Quantities:
    """A small helper class to store the different terms involved in evaluating
    the likelihood."""

    logp_a__d: Tensor
    log_p_d__a: Tensor
    log_p_d__a_no_perm: Tensor
    logp_a: Tensor
    logp_a_star: Tensor
    logp_a_star__d: Tensor


class InBatchGradients(Gradients):
    def __init__(self, eval_topk: Optional[int] = None, debug: bool = False, **kwargs):
        super().__init__(**kwargs)
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
            reader_score, retriever_score
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
        retrieval_score: Optional[Tensor] = None,
        **kwargs,
    ):

        # parameters
        reader_kl_weight = kwargs.get("reader_kl_weight", None)
        retriever_kl_weight = kwargs.get("retriever_kl_weight", None)

        # evaluate likelihood
        q = self.evaluate_likelihood_terms(
            retriever_score,
            reader_score,
            targets,
            eval_topk=self.eval_topk,
            debug=self.debug,
        )

        # compute the loss
        loss, diagnostics = self.compute_loss(
            q,
            targets=targets,
            reader_score=reader_score,
            retriever_score=retriever_score,
            retrieval_score=retrieval_score,
        )

        # run diagnostics
        diagnostics.update(
            retriever_diagnostics(
                retriever_score=retriever_score,
                retrieval_score=retrieval_score,
                reader_score=reader_score,
                **kwargs,
            )
        )

        # regularization
        kl_reader = batch_reduce(kl_divergence(q.logp_a, dim=1), op=torch.mean)
        diagnostics["reader/kl_uniform"] = kl_reader
        kl_retriever = batch_reduce(kl_divergence(retriever_score, dim=-1), op=torch.mean)
        diagnostics["retriever/kl_uniform"] = kl_retriever

        # auxiliary loss terms
        if reader_kl_weight is not None:
            loss = loss + reader_kl_weight * kl_reader
        if retriever_kl_weight is not None:
            loss = loss + retriever_kl_weight * kl_retriever

        # add the relevance targets for the retriever
        diagnostics.update(self._get_relevance_metrics(kwargs.get("match_score", None)))

        return {
            "loss": loss,
            "reader/logp": q.logp_a_star.detach(),
            "reader/entropy": -(q.logp_a.exp() * q.logp_a).sum(dim=1).mean().detach(),
            "_reader_scores_": reader_score.detach(),
            "_reader_logits_": q.logp_a.detach(),
            "_reader_targets_": targets.detach(),
            "_retriever_scores_": q.log_p_d__a_no_perm.detach(),
            "_retriever_reading_logits_": q.log_p_d__a_no_perm.sum(-1).detach(),
            **diagnostics,
        }

    def compute_loss(
        self, q: Quantities, *, targets: Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return -1 * (q.logp_a_star).mean(-1), {}
