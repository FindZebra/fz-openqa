import abc
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
        **kwargs,
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
    """Compute the gradients using Reinforce with importance sampling and priority sampling."""

    max_baseline_samples: int = 5
    baseline_dtype = None
    use_baseline: bool = False  # todo
    space: Space = Space.EXP

    def __call__(
        self,
        *,
        retriever_score: Tensor,
        reader_score: Tensor,
        targets: Tensor,
        retrieval_score: Optional[Tensor] = None,
        retrieval_log_prob: Optional[Tensor] = None,
        retrieval_log_Z: Optional[Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Compute the loss and diagnostics. This methods assumes using three probabilities
        distributions:
            1. reader: `p_\theta(ast|A, D) =  \exp f_\theta(D,ast) / \sum_A \exp f_\theta(D,A)`
            3. retriever: `p_\phi(D|A)` = softmax(`f_\phi(D,A)`) / \sum_D softmax(`f_\phi(D,A)`)
            3. checkpoint: `p_\psi(D|A)` = softmax(`f_\psi(D,A)`) / \sum_D softmax(`f_\psi(D,A)`)
            4. proposal: q(D)

        Parameters
        ----------
        retriever_score
            f_\phi(D): shape [bs, n_opts, n_docs]
        reader_score
            f_\theta(D): shape [bs, n_opts, n_docs]
        targets
            a_\star: shape [bs]
        retrieval_score
            f_\psi(D): shape [bs, n_opts, n_docs]
        retrieval_log_prob
            log q(D): shape [bs, n_opts, n_docs]
        retrieval_log_Z
            \log \sum_{D \in S} \exp f_\psi(D): shape [bs, n_opts, n_docs]

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary of diagnostics including the loss.

        """
        if self.space == Space.LOG:
            warnings.warn("ReinforceGradients has not been tested for log space")

        f_phi = retriever_score
        f_psi = retrieval_score
        log_Z_psi = retrieval_log_Z
        log_q = retrieval_log_prob
        diagnostics = {}

        # check inputs
        assert f_phi is not None
        assert f_psi is not None
        assert log_Z_psi is not None
        assert log_q is not None
        assert ((log_Z_psi.mean(-1) - log_Z_psi[..., 0]).abs() < 1e-3).all()
        log_Z_psi = log_Z_psi[..., :1]

        # scale the retriever scores to match the retrieval scores
        M_psi = f_psi.max(dim=-1, keepdim=True).values
        M_phi = f_phi.max(dim=-1, keepdim=True).values
        f_phi = f_phi - M_phi + M_psi

        with torch.no_grad():
            # `log \zeta = f_\phi(d) - f_\psi(d)`
            log_zeta = f_phi - f_psi

            # compute log p_psi(d, q, \Dset)
            log_p_psi = f_psi - log_Z_psi

            # `log \hat{\pi}_\phi(d | q, \Dset)`
            log_W_psi = log_p_psi - log_q
            log_Z_phi = log_Z_psi + (log_W_psi + log_zeta).logsumexp(dim=-1, keepdim=True)
            log_p_phi_hat = f_phi - log_Z_phi

            # `log w(d)`
            log_W = log_p_phi_hat - log_q

        # compute cartesian product: `D \in \Dset^{(M)}`
        reader_score_, f_phi_, log_W_, log_zeta_, log_W_psi_ = batch_cartesian_product(
            [reader_score, f_phi, log_W, log_zeta, log_W_psi]
        )
        # w = \prod_{j=1}^M W_j), for D in \Dset^{(M)}
        log_w_ = log_W_.sum(1)

        # importance weights diagnostics
        # effective sample size
        K = log_w_.size(-1)
        log_ess = 2 * log_w_.logsumexp(dim=-1) - (2 * log_w_).logsumexp(dim=-1)
        diagnostics["retriever/ess"] = log_ess.exp().mean()
        diagnostics["retriever/ess-ratio"] = log_ess.exp().mean() / K
        diagnostics["retriever/ess-ratio-std"] = log_ess.exp().std() / K
        diagnostics["retriever/w-max"] = log_w_.max().exp()
        diagnostics["retriever/w-std"] = log_w_.exp().std()

        # compute the reading likelihood estimate `p(A | q, A, S^{(M)})`
        log_p_A__D = reader_score_.log_softmax(dim=1)
        targets_ = targets.view(-1, 1, 1).expand(targets.shape[0], 1, log_p_A__D.shape[-1])
        log_p_ast__D = log_p_A__D.gather(1, targets_).squeeze(1)
        if self.space == Space.EXP:
            log_p_ast__S = (log_w_ + log_p_ast__D).logsumexp(-1)
        elif self.space == Space.LOG:
            log_p_ast__S = (log_w_.exp() + log_p_ast__D).sum(-1)
        else:
            raise ValueError(f"Unknown space: {self.space}")

        # compute loss for the retriever parameters
        if self.use_baseline:
            log_baseline = self.baseline(
                reader_score,
                targets=targets,
                log_W=log_W,
                max_samples=self.max_baseline_samples,
                space=self.space,
            )
        else:
            log_baseline = None
        # 1. compute the gradients using Reinforce
        h_log_weight = log_zeta_ - (log_W_psi_ + log_zeta_).logsumexp(-1, keepdim=True)
        h = f_phi_ - (h_log_weight.exp() * f_phi_).sum(-1, keepdim=True)
        with torch.no_grad():
            if self.space == Space.EXP:
                grad_weight = (log_w_ + log_p_ast__D).exp()
            elif self.space == Space.LOG:
                grad_weight = log_w_.exp() + log_p_ast__D
            else:
                raise ValueError(f"Unknown space: {self.space}")

            # 2. add the baseline weight
            if log_baseline is not None:
                grad_weight_baseline = (log_w_ + log_baseline).exp()
                diagnostics["retriever/baseline-l2"] = (
                    (grad_weight_baseline - grad_weight).pow(2).mean()
                )
                diagnostics["retriever/grad-weight-std"] = grad_weight.std(dim=-1).mean()
                diagnostics["retriever/grad-weight-controlled-std"] = (
                    (grad_weight_baseline - grad_weight).std(dim=-1).mean()
                )
            else:
                grad_weight_baseline = 0

        grad_log_p_ast__S = ((grad_weight - grad_weight_baseline).detach() * h.sum(1)).sum(-1)

        # compute the final loss
        loss = -1 * (log_p_ast__S + grad_log_p_ast__S).mean()

        # reader logits log p(A | q, A, S^{(M)})
        log_p_A__S = (log_w_.unsqueeze(1) + log_p_A__D).logsumexp(-1)

        return {
            "loss": loss,
            "reader/logp": log_p_ast__S.detach(),
            "_reader_logits_": log_p_A__S.detach(),
            "_reader_targets_": targets.detach(),
            "_doc_logits_": log_p_phi_hat.detach(),
            "_retriever_reading_logits_": (log_W * log_p_phi_hat).sum(-1).detach(),
            **diagnostics,
        }

    @staticmethod
    @torch.no_grad()
    def baseline(
        reader_scores: Tensor,
        *,
        targets: Tensor,
        log_W: Tensor = None,
        dtype: Optional[torch.dtype] = None,
        max_samples: int = 10,
        space: Space = None,
        **kwargs,
    ) -> Tensor:
        """Compute the controlled score for the given targets."""
        DEBUG = 0

        if DEBUG:
            rich.print(f">> baseline::reader_scores: {reader_scores.shape}, log_w:{log_W.shape}")

        if dtype is not None:
            original_dtype = reader_scores.dtype
            reader_scores = reader_scores.to(dtype)
        else:
            original_dtype = None
        if log_W is not None:
            log_W = log_W.to(dtype)

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

        def expand_and_index(scores, index):
            """Expands the scores of shape [ns, n_opts, n_docs] to scores
            of shape [bs, n_opts, n_docs, n_docs-1]"""
            if scores is None:
                return
            scores = scores.view(bs, n_opts, 1, n_docs).expand(bs, n_opts, n_docs, n_docs)
            return scores.gather(dim=3, index=index)

        _index = no_id_index.view(1, 1, *no_id_index.shape).expand(bs, n_opts, *no_id_index.shape)
        reader_scores = expand_and_index(reader_scores, _index)
        log_W = expand_and_index(log_W, _index)
        if DEBUG:
            rich.print(f">> scores::gather::1 {reader_scores.shape}")

        # M^K permutations of the scores across dim 3: the resulting tensors are of shape
        # [bs, n_opts, n_docs^M, n_docs-1]
        reader_scores, log_W = batch_cartesian_product([reader_scores, log_W])
        if DEBUG:
            log_mem_size(reader_scores, "retriever_scores::expand::1")
            log_mem_size(log_W, "log_W::expand::1")

        # truncate the `reader_scores` to the `top(max_samples)` according to the `retriever_scores`
        if reader_scores.shape[-1] > max_samples:
            _index = log_W.argsort(dim=-1, descending=True)[..., :max_samples]
            reader_scores = reader_scores.gather(dim=-1, index=_index)
            log_W = log_W.gather(dim=-1, index=_index)
            del _index

        if DEBUG:
            log_mem_size(reader_scores, "reader_scores::truncated::2")

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

        reader_scores, log_W = flatten_and_prod([reader_scores, log_W])
        if DEBUG:
            log_mem_size(reader_scores, "retriever_scores::expanded::3")

        if not reader_scores.is_cuda:
            # softmax not implemented for half precision on cpu
            reader_scores = reader_scores.float()
            log_W = log_W.float()

        # compute the baseline 1/N \sum_j p(a_star | q, A, D_j)
        log_p_A_D = reader_scores.log_softmax(dim=1)
        _targets = targets.view(bs, 1, 1, 1).expand(bs, 1, *log_p_A_D.shape[-2:])
        log_p_a_star_D = log_p_A_D.gather(dim=1, index=_targets).squeeze(1)

        # average the baseline over the documents
        log_w = log_W.sum(dim=1)
        if space == Space.EXP:
            log_sum_p_a_star_D = (log_w + log_p_a_star_D).logsumexp(dim=-1)
            baseline = log_sum_p_a_star_D - log_w.logsumexp(-1)
        elif space == Space.LOG:
            sum_log_p_a_star_D = (log_w.exp() + log_p_a_star_D).sum(dim=-1)
            baseline = sum_log_p_a_star_D - log_w.logsumexp(-1)
        else:
            raise ValueError(f"Unknown space: {space}")

        if original_dtype is not None:
            baseline.to(original_dtype)

        return baseline


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
