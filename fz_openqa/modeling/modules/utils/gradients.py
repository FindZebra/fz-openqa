import abc
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import einops
import numpy as np
import rich
import torch
from torch import Tensor
from torch.nn import functional as F

from .utils import batch_cartesian_product
from fz_openqa.datamodules.index.utils.io import log_mem_size
from fz_openqa.utils.pretty import get_separator
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
        pids: Optional[Tensor] = None,
    ):
        q = self.evaluate_likelihood_terms(
            retriever_score,
            reader_score,
            targets,
            eval_topk=self.eval_topk,
            debug=self.debug,
        )
        loss = self.compute_loss(q, targets=targets, retriever_score=retriever_score)
        return {
            "loss": loss,
            "reader/logp": q.logp_a_star.detach(),
            "_reader_logits_": q.logp_a.detach(),
            "_reader_targets_": targets.detach(),
            "_doc_logits_": q.log_p_d__a_no_perm.detach(),
            "_retriever_reading_logits_": q.log_p_d__a_no_perm.sum(-1).detach(),
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
    """Compute the gradients using Reinforce with leave-one-out baseline."""

    def compute_loss(
        self, q: Quantities, *, targets: Tensor, retriever_score: Tensor = None, **kwargs
    ) -> torch.Tensor:
        if retriever_score is None:
            raise ValueError(f"pids must be provided for {type(self).__name__}")

        # loss for the reader: E[ \sum_D \nabla p(q_star | q, A, D)]
        logp_a__d = q.logp_a__d.sum(2)
        logp_a_star__d = torch.gather(logp_a__d, dim=1, index=targets[:, None]).squeeze(1)
        p_a_star__d = logp_a_star__d.exp()
        reader_loss = -1 * p_a_star__d

        # loss for the retriever: REINFORCE
        use_baseline = True
        index = targets[:, None, None].expand(*targets.shape, 1, q.logp_a__d.shape[2])
        logp_a_star__D = torch.gather(q.logp_a__d, dim=1, index=index).squeeze(1)
        score = logp_a_star__D.exp()

        if not use_baseline:
            baseline = torch.zeros_like(score)
        else:
            rich.print(f">> expanded:shaped: {q.logp_a__d.shape}")
            baseline = self.baseline(retriever_score, targets, log_p_d__a=q.log_p_d__a)
            rich.print(f"> baseline: {baseline.shape}, score:{score.shape}")

        log_p_D_A = q.log_p_d__a.sum(1)
        controlled_score = (score.unsqueeze(-1) - baseline).detach()
        retriever_loss = -1 * (controlled_score * log_p_D_A).sum(dim=(-1))

        rich.print(
            f">> retriever_loss: {retriever_loss.shape}, " f"reader_loss: {reader_loss.shape}"
        )
        rich.print(
            f">> exp: Z_diff: {controlled_score.mean(-1)}, "
            f"logp_a_star__D: {logp_a_star__D.mean(-1)}"
        )
        with torch.no_grad():
            for j in range(controlled_score.shape[1]):
                rich.print(
                    f"> Z_diff: {controlled_score[0, j]}, "
                    f"x: {score[0]}, "
                    f"x:hat: {(baseline)[0, j]}"
                )

        return reader_loss + retriever_loss

    @staticmethod
    @torch.no_grad()
    def baseline(retriever_score: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute the controlled score for the given targets."""

        # todo: debug
        idx = torch.arange(0, retriever_score.numel())
        retriever_score = idx.view(retriever_score.shape)

        bs, n_opts, n_docs = retriever_score.shape
        m_docs = (n_opts - 1) ** n_docs

        # generate an index of size [n_docs, n_docs-1] where
        #  each row `i` is contains indices: [0 ...i-1 ... i+1 ... n_docs-1]
        # https://discuss.pytorch.org/t/keep-off-diagonal-elements-only-from-square-matrix/54379
        no_id_index = torch.arange(0, n_opts, 1, device=retriever_score.device, dtype=torch.long)
        no_id_index = no_id_index.view(1, n_opts).expand(n_opts, n_opts)
        no_id_index = (
            no_id_index.flatten()[1:]
            .view(n_opts - 1, n_opts + 1)[:, :-1]
            .reshape(n_opts, n_opts - 1)
        )
        rich.print(f">> no_id_index: {no_id_index}")

        # get a matrix of score of shape [bs, n_opts, n_docs, n_docs-1]
        rich.print(f">> input_scores: {retriever_score.shape}")
        scores = retriever_score.view(bs, 1, n_opts, n_docs).expand(bs, n_opts, n_opts, n_docs)
        _index = no_id_index.view(1, *no_id_index.shape, 1).expand(bs, *no_id_index.shape, n_docs)
        scores = scores.gather(dim=2, index=_index)
        rich.print(f">> scores: {scores.shape}")

        # compute the score permutations for the `n_docs-1`
        in_pattern = "bs n_opts m_opts x_docs"
        out_pattern = "(bs n_opts) m_opts x_docs"
        scores = einops.rearrange(scores, f"{in_pattern} -> {out_pattern}")
        rich.print(f">> flat::scores: {scores.shape}")
        scores, *_ = batch_cartesian_product([scores])
        scores = einops.rearrange(scores, f"{out_pattern} -> {in_pattern}", bs=bs)
        rich.print(f">> perm::scores: {scores.shape}")

        # add the idenity to the score
        _scores = -1 * torch.ones(
            bs, n_opts, n_opts, m_docs, dtype=scores.dtype, device=scores.device
        )
        _index = no_id_index.view(1, *no_id_index.shape, 1).expand(bs, *no_id_index.shape, m_docs)
        rich.print(f">> _index: {_index.shape}")
        _scores.scatter_(dim=2, index=_index, src=scores)

        rich.print(f">> 1:_scores: {_scores.shape}")
        rich.print(_scores[0])

        rich.print(f">>>>> _scores: {_scores.shape}, retriever_score: {retriever_score.shape}")

        _index = torch.arange(0, n_opts, 1, device=retriever_score.device, dtype=torch.long)
        rich.print(f"# index: {_index}")
        _index = _index.view(1, 1, n_opts, 1, 1).expand(bs, n_opts, n_opts, n_docs, m_docs)
        src = retriever_score.view(bs, 1, n_opts, n_docs, 1).repeat(1, n_opts, 1, 1, 1)
        src = torch.repeat_interleave(src, m_docs, dim=-1)
        _scores = _scores.view(bs, n_opts, n_opts, 1, m_docs).repeat(1, 1, 1, n_docs, 1)
        # src = -7 * torch.ones_like(src)
        rich.print(f">>> src: {src.shape}, _index:{_index.shape}, _scores: {_scores.shape}")
        _scores.scatter_(dim=1, index=_index, src=src)
        _scores = einops.rearrange(_scores, "... n_docs m_docs -> ... (n_docs m_docs)")

        rich.print(f">> 2:_scores: {_scores.shape}")
        rich.print(_scores[0, :])
        print(get_separator())
        # rich.print(src[0, 0])
        # print(get_separator())
        rich.print(retriever_score[0])

        exit()

        # compute the baseline
        log_p_A_D = scores.log_softmax(dim=1)
        _targets = targets.view(bs, 1, 1, 1).expand(bs, n_opts, n_docs, m_docs)
        log_p_a_star_D = log_p_A_D.gather(dim=1, index=_targets)
        p_a_star_D = log_p_a_star_D.exp()
        return p_a_star_D.mean(dim=-1)

        exit()

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
