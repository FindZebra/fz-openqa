import math
import warnings
from typing import Dict
from typing import List
from typing import Optional

import einops
import rich
import torch
from torch import Tensor

from fz_openqa.datamodules.index.utils.io import log_mem_size
from fz_openqa.modeling.gradients.base import Gradients
from fz_openqa.modeling.gradients.base import Space
from fz_openqa.modeling.gradients.utils import batch_cartesian_product


def kl_uniform(logits: Tensor, dim: int = -1) -> Tensor:
    log_p = logits.log_softmax(dim=dim)
    N = log_p.size(dim)
    log_q = -torch.ones_like(log_p) * math.log(N)
    return (log_p.exp() * (log_p - log_q)).sum(dim=dim)


class ReinforceGradients(Gradients):
    def __init__(
        self,
        *args,
        baseline_dtype: Optional[torch.dtype] = torch.float16,
        use_baseline: bool = True,
        w_max: Optional[float] = None,
        space: Space = Space.EXP,
        max_baseline_samples: int = 3,
        expr: str = "B",
        gamma: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.space = Space(space)
        self.baseline_dtype = baseline_dtype
        self.use_baseline = use_baseline
        self.log_w_max = math.log(w_max) if w_max is not None else float("inf")
        self.max_baseline_samples = max_baseline_samples
        self.expr = expr
        self.gamma = gamma

        rich.print(
            f"{self.__class__.__name__}: "
            f"use_baseline={self.use_baseline}, "
            f"expr={self.expr}, "
            f"gamma={self.gamma}"
        )

    def __call__(
        self,
        *,
        retriever_score: Tensor,
        reader_score: Tensor,
        targets: Tensor,
        retrieval_score: Optional[Tensor] = None,
        retrieval_log_weight: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Compute the loss and diagnostics. This methods assumes using three probabilities
        distributions:
            1. reader: `p_\theta(ast|A, D) =  \exp f_\theta(D,ast) / \sum_A \exp f_\theta(D,A, q)`
            3. retriever: `p_\phi(d|q)` = softmax(`f_\phi(d,q)`) / \sum_D softmax(`f_\phi(d,q)`)
            3. checkpoint: `p_\psi(d|q)` = softmax(`f_\psi(d,q)`) / \sum_D softmax(`f_\psi(d,q)`)
            4. proposal: q(d) (priority sampling), with importance weights s(d) = p_\psi(d|q) / q(d)

        Parameters
        ----------
        retriever_score
            f_\phi(d,q): shape [bs, n_opts, n_docs]
        reader_score
            f_\theta(d,q): shape [bs, n_opts, n_docs]
        targets
            a_\star: shape [bs]
        retrieval_score
            f_\psi(d,q): shape [bs, n_opts, n_docs]
        retrieval_log_weight
            log s(d,q): shape [bs, n_opts, n_docs]

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary of diagnostics including the loss.

        """
        gamma = kwargs.get("gamma", self.gamma)
        reader_kl_weight = kwargs.get("reader_kl_weight", None)
        retriever_kl_weight = kwargs.get("retriever_kl_weight", None)

        f_theta_ = reader_score
        f_phi_ = retriever_score
        f_psi_ = retrieval_score
        log_s_ = retrieval_log_weight
        diagnostics = {}

        # check inputs
        assert f_theta_ is not None
        assert f_phi_ is not None
        assert f_psi_ is not None
        assert log_s_ is not None

        # reader likelihood `log p(a | d, q)`
        log_p_a__d_ = f_theta_.log_softmax(dim=1)

        # `log \zeta = f_\phi(d) - f_\psi(d)`
        log_zeta_ = f_phi_ - f_psi_

        # `log \hat{\zeta} = \log p(a | d, q) + log \zeta`
        log_zeta_hat_ = log_zeta_ + log_p_a__d_

        # importance weights
        log_w_ = (log_s_ + log_zeta_).log_softmax(dim=-1).detach()
        log_w_hat_ = (log_s_ + log_zeta_hat_).log_softmax(dim=-1).detach()

        # '\nabla p(d | q, a) = `\nabla f_\phi(d) - sum_{d \in S} w(d) \nabla f_\phi(d)`
        # log_p_d__a_ = f_phi - (log_w_ * f_phi).logsumexp(dim=-1, keepdim=True)
        log_p_d__a_ = f_phi_ - (log_w_.exp().detach() * f_phi_).sum(dim=-1, keepdim=True)

        # compute cartesian product: `D \in \Dset^{(M)}`
        args = f_theta_, f_phi_, f_psi_, log_s_, log_w_, log_w_hat_, log_p_d__a_
        f_theta, f_phi, f_psi, log_s, log_w, log_w_hat, log_p_d__a = batch_cartesian_product(*args)

        # reader likelihood `log p(a | d, q)`
        log_p_a__d = f_theta.log_softmax(dim=1)

        # retriever likelihood
        log_p_D__A = log_p_d__a.sum(1)

        # W(D) = \prod_{j=1}^M w_j), for D in \Dset^{(M)}
        log_W = log_w.sum(dim=1)
        log_W_hat = log_w_hat.sum(dim=1)
        self.ess_diagnostics(diagnostics, log_W)

        # compute the log-likelihood estimate
        log_p_a = (log_W.unsqueeze(1) + log_p_a__d).logsumexp(dim=-1)
        log_p_ast = log_p_a.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        lb_p_a = (log_W.unsqueeze(1) + log_p_a__d).sum(dim=-1)
        lb_p_ast = lb_p_a.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        diagnostics["reader/lb"] = lb_p_ast
        diagnostics["reader/kl_lb"] = log_p_ast - lb_p_ast

        # compute KL divergence w.r.t to a uniform prior (regularization)
        kl_reader = kl_uniform(log_p_a, dim=1)
        diagnostics["reader/kl_uniform"] = kl_reader
        kl_retriever = kl_uniform(f_phi_, dim=2).sum(1)
        diagnostics["retriever/kl_uniform"] = kl_retriever

        # log p(a_st | q, A, D)
        targets_ = targets.view(targets.size(0), 1, 1)
        targets_ = targets_.expand(targets.size(0), 1, log_p_a__d.size(2))
        log_p_ast_d = log_p_a__d.gather(dim=1, index=targets_).squeeze(1)

        # compute the baseline (unused in current experiments)
        if self.use_baseline:
            space = {"A": Space.EXP, "B": Space.EXP, "C": Space.LOG}[self.expr]
            log_b = self.baseline(
                reader_scores=reader_score,
                log_w=log_w,
                targets=targets,
                space=space,
                dtype=self.baseline_dtype,
                max_samples=self.max_baseline_samples,
            )
        else:
            log_b = None

        # compute the gradient estimate
        if self.expr == "A":
            h = log_p_ast_d + gamma * log_p_D__A
            score = (log_W - log_p_ast.unsqueeze(-1) + log_p_ast_d).exp()
            if log_b is not None:
                score = score - (log_W - log_p_ast.unsqueeze(-1) + log_b).exp()
            loss = -1 * (score.detach() * h).sum(-1)

        elif self.expr == "A2":
            h = log_p_ast_d + gamma * log_p_D__A
            score = (log_W_hat).exp()
            if log_b is not None:
                raise NotImplementedError("Baseline not implemented for A2")
            loss = -1 * (score.detach() * h).sum(-1)

        elif self.expr in {"B", "B-zero"}:
            weight = (log_W + log_p_ast_d - log_p_ast.unsqueeze(-1)).exp()
            if log_b is not None:
                weight = weight - (log_W + log_b - log_p_ast.unsqueeze(-1)).exp()
            reader_loss = log_p_ast
            if self.expr == "B":
                retriever_loss = (weight.detach() * log_p_D__A.sum(1)).sum(-1)
            else:
                retriever_loss = 0
            loss = -(reader_loss + gamma * retriever_loss)

        elif self.expr == "C":
            if log_b is None:
                log_b = 0
            reader_weight = log_W.exp()
            retriever_weight = log_W.exp() * (log_p_ast_d - log_b)
            reader_loss = (reader_weight.detach() * log_p_ast_d).sum(-1)
            retriever_loss = (retriever_weight.detach() * log_p_D__A).sum(-1)
            loss = -(reader_loss + gamma * retriever_loss)
        else:
            raise ValueError(f"expr must be either A, B or C, got {self.expr}")

        # add the KL temrs
        if reader_kl_weight is not None:
            loss = loss + reader_kl_weight * kl_reader
        if retriever_kl_weight is not None:
            loss = loss + retriever_kl_weight * kl_retriever

        return {
            "loss": loss,
            "reader/entropy": -(log_p_a.exp() * log_p_a).sum(dim=1).mean().detach(),
            "reader/logp": log_p_ast.detach(),
            "_reader_logits_": log_p_a.detach(),
            "_reader_scores_": reader_score.detach(),
            "_reader_targets_": targets.detach(),
            "_retriever_scores_": retriever_score.detach(),
            "_retriever_reading_logits_": retriever_score.sum(-1).detach(),
            **diagnostics,
        }

    @staticmethod
    @torch.no_grad()
    def ess_diagnostics(diagnostics, log_W_):
        K = log_W_.size(-1)
        log_ess = 2 * log_W_.logsumexp(dim=-1) - (2 * log_W_).logsumexp(dim=-1)
        diagnostics["ess/mean"] = log_ess.exp().mean()
        diagnostics["ess/ratio-mean"] = log_ess.exp().mean() / K
        diagnostics["ess/max"] = log_W_.max().exp()

    @staticmethod
    @torch.no_grad()
    def baseline(
        reader_scores: Tensor,
        *,
        targets: Tensor,
        log_w: Tensor = None,
        dtype: Optional[torch.dtype] = None,
        max_samples: int = 10,
        space: Space = None,
        **kwargs,
    ) -> Tensor:
        """Compute the controlled score for the given targets."""
        DEBUG = 0

        if DEBUG:
            rich.print(f">> baseline::reader_scores: {reader_scores.shape}, log_w:{log_w.shape}")

        if dtype is not None:
            original_dtype = reader_scores.dtype
            reader_scores = reader_scores.to(dtype)
        else:
            original_dtype = None
        if log_w is not None:
            log_w = log_w.to(dtype)

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
        log_w = expand_and_index(log_w, _index)
        if DEBUG:
            rich.print(f">> scores::gather::1 {reader_scores.shape}")

        # M^K permutations of the scores across dim 3: the resulting tensors are of shape
        # [bs, n_opts, n_docs^M, n_docs-1]
        reader_scores, log_w = batch_cartesian_product(reader_scores, log_w)
        if DEBUG:
            log_mem_size(reader_scores, "retriever_scores::expand::1")
            log_mem_size(log_w, "log_W::expand::1")

        # truncate the `reader_scores` to the `top(max_samples)` according to the `retriever_scores`
        if reader_scores.shape[-1] > max_samples:
            _index = log_w.argsort(dim=-1, descending=True)[..., :max_samples]
            reader_scores = reader_scores.gather(dim=-1, index=_index)
            log_w = log_w.gather(dim=-1, index=_index)
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
            scores = batch_cartesian_product(*scores)
            scores = [
                einops.rearrange(s, f"{out_pattern} -> {in_pattern}", bs=bs)
                if s is not None
                else None
                for s in scores
            ]
            return scores

        reader_scores, log_w = flatten_and_prod([reader_scores, log_w])
        if DEBUG:
            log_mem_size(reader_scores, "retriever_scores::expanded::3")

        if not reader_scores.is_cuda:
            # softmax not implemented for half precision on cpu
            reader_scores = reader_scores.float()
            log_w = log_w.float()

        # compute the baseline 1/N \sum_j p(a_star | q, A, D_j)
        log_p_A_D = reader_scores.log_softmax(dim=1)
        _targets = targets.view(bs, 1, 1, 1).expand(bs, 1, *log_p_A_D.shape[-2:])
        log_p_a_star_D = log_p_A_D.gather(dim=1, index=_targets).squeeze(1)

        # average the baseline over the documents
        log_W = log_w.sum(dim=1)
        log_W_normalized = log_W - log_W.logsumexp(dim=-1, keepdim=True)
        if space == Space.EXP:
            log_sum_p_a_star_D = (log_W_normalized + log_p_a_star_D).logsumexp(dim=-1)
            baseline = log_sum_p_a_star_D
        elif space == Space.LOG:
            sum_log_p_a_star_D = (log_W_normalized.exp() + log_p_a_star_D).sum(dim=-1)
            baseline = sum_log_p_a_star_D
        else:
            raise ValueError(f"Unknown space: {space}")

        if original_dtype is not None:
            baseline.to(original_dtype)

        return baseline
