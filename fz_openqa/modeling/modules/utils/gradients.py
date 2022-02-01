import abc
import math
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import einops
import numpy as np
import rich
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from .utils import batch_cartesian_product
from fz_openqa.datamodules.index.utils.io import log_mem_size
from fz_openqa.utils.pretty import pprint_batch


@torch.no_grad()
def plot_scores(scores, controlled_scores):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set()
    colors = sns.color_palette()

    rich.print(f">> scores: {scores.shape}")
    rich.print(f">> controlled_scores: {controlled_scores.shape}")

    scores = scores.detach().cpu().numpy()
    controlled_scores = controlled_scores.detach().cpu().numpy()

    for k in range(scores.shape[0]):
        rich.print(f">> score: {scores[k]}, controlled_score: {controlled_scores[k]}")

    sns.distplot(scores, bins=10, color=colors[0], label="score")
    sns.distplot(controlled_scores, bins=10, color=colors[1], label="controlled")
    plt.axvline(x=np.mean(scores), color=colors[0])
    plt.axvline(x=np.mean(controlled_scores), color=colors[1])
    plt.legend()
    plt.show()


class GradExpression(Enum):
    VARIATIONAL = "variational"
    IN_BATCH = "in_batch"
    REINFORCE = "reinforce"


class Space(Enum):
    EXP = "exp"
    LOG = "log"


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


class Estimator(nn.Module):
    def __init__(self, eval_topk: Optional[int] = None, debug: bool = False):
        super().__init__()
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
    ):
        q = self.evaluate_likelihood_terms(
            retriever_score,
            reader_score,
            targets,
            eval_topk=self.eval_topk,
            debug=self.debug,
        )
        loss, diagnostics = self.compute_loss(
            q,
            targets=targets,
            reader_score=reader_score,
            retriever_score=retriever_score,
            retrieval_score=retrieval_score,
        )
        return {
            "loss": loss,
            "reader/logp": q.logp_a_star.detach(),
            "_reader_logits_": q.logp_a.detach(),
            "_reader_targets_": targets.detach(),
            "_doc_logits_": q.log_p_d__a_no_perm.detach(),
            "_retriever_reading_logits_": q.log_p_d__a_no_perm.sum(-1).detach(),
            **diagnostics,
        }

    @abc.abstractmethod
    def compute_loss(
        self, q: Quantities, *, targets: Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        ...


class InBatchGradients(Estimator):
    """Compute the gradients of the option retriever assuming the current
    batch to be the whole dataset."""

    def compute_loss(
        self, q: Quantities, *, targets: Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return -1 * (q.logp_a_star).mean(-1), {}


class VariationalGradients(Estimator):
    """Compute the gradients using a Variational Lower Bound."""

    def compute_loss(
        self, q: Quantities, *, targets: Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        lb_logp_a = (q.logp_a__d + q.log_p_d__a.sum(dim=1, keepdim=True)).sum(dim=2)
        lb_logp_a_star = torch.gather(lb_logp_a, dim=1, index=targets[:, None]).squeeze(1)
        return -1 * (lb_logp_a_star).mean(-1), {}


class ReinforceGradients(Estimator):
    """Compute the gradients using Reinforce with leave-one-out baseline and importance-weighting"""

    max_baseline_samples: int = 5
    baseline_dtype = None
    space: Space = Space.LOG

    def compute_loss(
        self,
        q: Quantities,
        *,
        targets: Tensor,
        reader_score: Tensor = None,
        retriever_score: Tensor = None,
        retrieval_score: Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if reader_score is None:
            raise ValueError(f"reader_score must be provided for {type(self).__name__}")

        diagnostics = {}
        use_baseline = True
        use_importance_weights = True
        iw_expr = 2
        if retrieval_score is None:
            retrieval_score = retriever_score.detach()

        # expand retriever and retrieval scores
        retriever_score_ = retriever_score
        retriever_score, retrieval_score = batch_cartesian_product(
            [retriever_score, retrieval_score]
        )

        # compute log_zeta
        log_zeta = retriever_score - retrieval_score
        log_N = math.log(log_zeta.shape[-1])
        log_E_zeta = log_zeta.logsumexp(dim=-1, keepdim=True) - log_N

        # compute the importance weights
        if use_importance_weights and self.training:
            with torch.no_grad():
                if iw_expr == 1:
                    log_w = log_zeta - log_E_zeta
                elif iw_expr == 2:
                    log_w = retriever_score.log_softmax(dim=-1) - retrieval_score.log_softmax(
                        dim=-1
                    )
                else:
                    raise ValueError(f"Invalid iw_expr={iw_expr}")
                log_w = log_w.sum(1)
                w = log_w.exp()
                diagnostics["retriever/w-max"] = w.max()

                # truncate
                # rich.print(f"> w: {w.min()} : {w.max()} ({w.mean()})")
                # w = w.clamp(max=1e2)

                # effective sample size
                ess = w.sum(-1).pow(2) / w.pow(2).sum(-1)
                ess = ess.mean()
                diagnostics["retriever/ess"] = ess

                # rich.print(f"> ESS: {ess}")
        else:
            w = torch.ones_like(
                log_zeta[
                    :,
                    0,
                ]
            )

        index = targets[:, None, None].expand(*targets.shape, 1, q.logp_a__d.shape[2])
        logp_a_star__D = torch.gather(q.logp_a__d, dim=1, index=index).squeeze(1)

        # loss for the reader: E[ \sum_D \nabla p(q_star | q, A, D)]
        if self.space == Space.EXP:
            reader_loss = -1 * (w * logp_a_star__D.exp()).sum(-1)
        elif self.space == Space.LOG:
            reader_loss = -1 * (w * logp_a_star__D).sum(-1)
        else:
            raise ValueError("space must be either 'exp' or 'log'")

        # loss for the retriever: REINFORCE
        if self.space == Space.EXP:
            # todo: use log sum exp trick to compute log of the sum of exp
            score = logp_a_star__D.exp().detach()
        elif self.space == Space.LOG:
            score = logp_a_star__D.detach()
        else:
            raise ValueError("space must be either 'exp' or 'log'")

        if use_baseline and self.training:
            # rich.print(f">> expanded:shaped: {q.logp_a__d.shape}")
            baseline = self.baseline(
                reader_score,
                targets=targets,
                retriever_scores=retriever_score_,
                max_samples=self.max_baseline_samples,
                dtype=self.baseline_dtype,
                space=self.space,
            )
            # rich.print(f"> baseline: {baseline.shape}, score:{score.shape}")
        else:
            baseline = 0

        # compute an estimate of log_p_D_A
        # normalizer = (log_zeta - log_E_zeta).exp().detach() * log_zeta
        # log_p_D_A = retriever_score - normalizer
        # log_p_D_A = log_p_D_A.sum(1)
        log_p_D_A = q.log_p_d__a.sum(1)

        # compute the final loss
        controlled_score = (w * (score - baseline)).detach()
        retriever_loss = -1 * (controlled_score * log_p_D_A).sum(dim=(-1))

        # plot_scores(score[0], controlled_score[0])
        # time.sleep(3)

        return reader_loss + retriever_loss, diagnostics

    @staticmethod
    @torch.no_grad()
    def importance_weights(log_p_d__a: Tensor, retrieval_score: Tensor) -> Tensor:

        # clip the retrieval score for stability
        M = retrieval_score.max(dim=-1).values.unsqueeze(-1)
        retrieval_score = retrieval_score - M
        retrieval_score = retrieval_score.clip_(min=-1e3)

        # compute the permutations of the retrieval scores for each D in D^(M)
        retrieval_score, *_ = batch_cartesian_product([retrieval_score])

        # compute log q(D|A)
        normalizer = retrieval_score.logsumexp(dim=2, keepdim=True)
        log_q_d__a = retrieval_score - normalizer

        # product over options
        log_q_d__a = log_q_d__a.sum(dim=1)
        log_p_d__a = log_p_d__a.sum(dim=1)

        # return w = p(D | A) / q(D | A)
        log_w = log_p_d__a - log_q_d__a
        w = log_w.exp()

        # clip for stability
        w = w.clip(max=10)

        return w

    @staticmethod
    @torch.no_grad()
    def baseline(
        reader_scores: Tensor,
        *,
        targets: Tensor,
        retriever_scores: Tensor = None,
        dtype: Optional[torch.dtype] = None,
        max_samples: int = 10,
        space: Space = Space.LOG,
        **kwargs,
    ) -> Tensor:
        """Compute the controlled score for the given targets."""
        DEBUG = False
        if max_samples >= reader_scores.shape[-1]:
            retriever_scores = None
        if dtype is not None:
            original_dtype = reader_scores.dtype
            reader_scores = reader_scores.to(dtype)
        else:
            original_dtype = None
        if retriever_scores is not None:
            retriever_scores = retriever_scores.to(dtype)

        bs, n_opts, n_docs = reader_scores.shape

        # generate an index of size [n_docs, n_docs-1] where
        #  each row `i` is contains indices: [0 ...i-1 ... i+1 ... n_docs-1]
        # https://discuss.pytorch.org/t/keep-off-diagonal-elements-only-from-square-matrix/54379
        no_id_index = torch.arange(0, n_docs, 1, device=reader_scores.device, dtype=torch.long)
        no_id_index = no_id_index.view(1, n_docs).expand(n_docs, n_docs)
        no_id_index = (
            no_id_index.flatten()[1:]
            .view(n_docs - 1, n_docs + 1)[:, :-1]
            .reshape(n_docs, n_docs - 1)
        )

        # rich.print(f">> no_id_index: {no_id_index}")

        def expand_and_index(scores, index):
            """Expands the scores of shape [ns, n_opts, n_docs] to scores
            of shape [bs, n_opts, n_docs, n_docs-1]"""
            if scores is None:
                return
            scores = scores.view(bs, n_opts, 1, n_docs).expand(bs, n_opts, n_docs, n_docs)
            return scores.gather(dim=3, index=index)

        _index = no_id_index.view(1, 1, *no_id_index.shape).expand(bs, n_opts, *no_id_index.shape)
        reader_scores = expand_and_index(reader_scores, _index)
        retriever_scores = expand_and_index(retriever_scores, _index)
        if DEBUG:
            rich.print(f">> scores::gather::1 {reader_scores.shape}")

        # M^K permutations of the scores across dim 3: the resulting tensors are of shape
        # [bs, n_opts, n_docs^M, n_docs-1]
        reader_scores, retriever_scores = batch_cartesian_product([reader_scores, retriever_scores])
        if DEBUG:
            log_mem_size(reader_scores, "retriever_scores::expand::1")
            log_mem_size(retriever_scores, "retriever_scores::expand::1")

        # truncate the `reader_scores` to the `top(max_samples)` according to the `retriever_scores`
        if retriever_scores is not None:
            _index = retriever_scores.argsort(dim=-1, descending=True)[..., :max_samples]
            reader_scores = reader_scores.gather(dim=-1, index=_index)
            del _index, retriever_scores

        if DEBUG:
            log_mem_size(reader_scores, "retriever_scores::truncated::2")

        def flatten_and_prod(scores: List[Tensor]):
            """permutations of the dimension -1 of K-1 scores, resulting in
            a matrix of shape [bs, n_opts, n_docs^M, (n_docs-1)^M]"""
            in_pattern = "bs n_opts m_docs x_docs"
            out_pattern = "(bs m_docs) n_opts x_docs"

            scores = [
                einops.rearrange(s, f"{in_pattern} -> {out_pattern}") if s is not None else None
                for s in scores
            ]
            scores = batch_cartesian_product(scores)
            scores = [
                einops.rearrange(s, f"{out_pattern} -> {in_pattern}", bs=bs)
                if s is not None
                else None
                for s in scores
            ]
            return scores

        reader_scores, *_ = flatten_and_prod([reader_scores])
        if DEBUG:
            log_mem_size(reader_scores, "retriever_scores::expanded::3")

        if not reader_scores.is_cuda:
            # softmax not implemented for half precision on cpu
            reader_scores = reader_scores.float()

        # compute the baseline 1/N \sum_j p(a_star | q, A, D_j)
        log_p_A_D = reader_scores.log_softmax(dim=1)
        _targets = targets.view(bs, 1, 1, 1).expand(bs, 1, *log_p_A_D.shape[-2:])
        log_p_a_star_D = log_p_A_D.gather(dim=1, index=_targets).squeeze(1)

        if space == Space.EXP:
            N = log_p_a_star_D.shape[-1]
            log_sum_p_a_star_D = log_p_a_star_D.logsumexp(dim=-1)
            baseline = log_sum_p_a_star_D / N
        elif space == Space.LOG:
            baseline = log_p_a_star_D.mean(dim=-1)
        else:
            raise ValueError(f"Unknown space: {space}")

        if original_dtype is not None:
            baseline.to(original_dtype)

        return baseline

    @staticmethod
    @torch.no_grad()
    def __expand_version_score_estimate(scores: Tensor, *, pids: Tensor) -> Tensor:

        DEBUG = 100

        # repeat the values across the dimension -2, so the resulting tensors
        #  have shape [... n_docs, n_docs]
        def repeat(x, n=None):
            n = n or x.shape[-1]
            return x.unsqueeze(-2).expand(*x.shape[:-1], n, x.shape[-1])

        # repeat the documents across the dimension -1, so the resulting tensors is of
        # shape [bs(k), n_options(j), n_docs(l), n_docs(p)]
        _pids = repeat(pids)
        _scores = repeat(scores)

        # build the mask such that `d_l != d_p`
        mask = pids.unsqueeze(-1) != _pids

        # replace the ids of the masked documents with `-1`
        _pids = _pids.masked_fill(~mask, -1)
        _pids, sort_idx = _pids.sort(dim=-1)
        _scores = _scores.gather(dim=-1, index=sort_idx)
        mask = mask.gather(dim=-1, index=sort_idx)

        # remove the as many `-1`s as possible
        M = _pids.max() + 1
        _pids[_pids < 0] = M
        pad_idx = _pids.argmin(dim=-1).min()
        _pids[_pids == M] = -1
        _pids = _pids[..., pad_idx:]
        _scores = _scores[..., pad_idx:]
        mask = mask[..., pad_idx:]

        # It is expected to have the same number non-masked elements per row
        assert torch.all(mask >= 0)
        del mask

        if DEBUG > 5:
            k = 0
            for l_ in range(pids.shape[-1]):
                for j in range(pids.shape[1]):
                    rich.print(
                        f"\t- l={l_}, d_l: {pids[k, j, l_]}: d'_lp: {_pids[k, j, l_].tolist()}"
                    )

        del _pids
        # compute the permutations of the scores across [n_opts, n_docs(l)]
        _scores, *_ = batch_cartesian_product([_scores])

        if DEBUG > 0:
            log_mem_size(_scores, "_scores (expand 1)")
            # log_mem_size(mask, "mask (expand 1)")

        # compute the permutations of the scores across [n_opts, n_docs(l)]
        bs = _scores.shape[0]
        pattern = "bs n_options n_docs n_samples -> (bs n_docs) n_options n_samples"
        _scores = einops.rearrange(_scores, pattern)
        # mask = einops.rearrange(mask, pattern)
        _scores, *_ = batch_cartesian_product([_scores])
        reorder_pattern = "(bs n_docs) n_options n_samples -> bs n_options n_docs n_samples"
        _scores = einops.rearrange(_scores, reorder_pattern, bs=bs)
        # mask = einops.rearrange(mask, reorder_pattern, bs=bs)

        if DEBUG > 0:
            log_mem_size(_scores, "_scores (expand 2)")
            # log_mem_size(mask, "mask (expand 2)")

        # compute the final score estimate for each document and return
        # mask = mask.prod(dim=1)
        rich.print(f"> scores: {_scores[0]}")
        _scores = _scores.sum(dim=1)
        n_els = _scores.shape[-1]
        score_estimate = 1 / n_els * _scores.sum(dim=-1)

        if torch.isnan(score_estimate).any():
            path = "nan_score_estimate.npy"
            np.save(path, score_estimate.cpu().numpy())
            rich.print(f">> {score_estimate}")
            raise ValueError(f"NaN detected, n={torch.isnan(score_estimate).long().sum()}")

        if DEBUG > 5:
            k = 0
            scores, *_ = batch_cartesian_product([scores])
            scores = scores.sum(1)
            diff = scores - score_estimate
            for l_ in range(scores.shape[-1]):
                rich.print(
                    f"\t- l={l_}, diff={diff[k, l_]}, score={scores[k, l_]}, "
                    f"estimate={score_estimate[k, l_]}, n_els={n_els[k, l_]}"
                )

        if DEBUG > 50:
            import seaborn as sns
            import matplotlib.pyplot as plt

            sns.set()
            colors = sns.color_palette()
            sns.distplot(scores[k].detach().cpu().numpy(), color=colors[0], label="score")
            sns.distplot(diff[k].detach().cpu().numpy(), color=colors[1], label="controlled")
            plt.axvline(x=scores[k].mean(), color=colors[0])
            plt.axvline(x=diff[k].mean(), color=colors[1])
            plt.legend()
            plt.show()

        return score_estimate


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
