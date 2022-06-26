from typing import Dict
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from fz_openqa.modeling.gradients.base import Gradients
from fz_openqa.modeling.gradients.retriever_diagnostics import retriever_diagnostics
from fz_openqa.modeling.gradients.utils import batch_cartesian_product
from fz_openqa.modeling.gradients.utils import kl_divergence


def cleanup_nans(score, key):
    nan_count = score.isnan().float().sum()
    if nan_count > 0:
        no_nans = score[~score.isnan()]

        if no_nans.numel() > 0:
            score_mean = no_nans.mean().detach()
            non_nan_min = no_nans.min()
            non_nan_max = no_nans.max()
        else:
            score_mean = 0
            non_nan_min = -0
            non_nan_max = 0

        logger.warning(
            f"> {key:10s}: {nan_count} NaNs "
            f"({nan_count / score.numel():.2%}), "
            f"mean={score_mean:.1f} "
            f"range=[{non_nan_min:.1f}-{non_nan_max:.1f}]"
        )

        # replace NaNs
        score[score.isnan()] = score_mean

    return nan_count


def format_parameter(x):
    if isinstance(x, Tensor):
        unique_x = x.unique()
        if len(unique_x) != 1:
            raise ValueError(f"Parameter {x} has multiple values")
        return unique_x[0]
    else:
        return x


class RenyiGradients(Gradients):
    required_features = ["answer.target", "question.id", "question.loc"]

    def __init__(
        self,
        *args,
        cartesian_max_size: int = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cartesian_max_size = cartesian_max_size

    def __call__(
        self,
        *,
        retriever_score: Tensor = None,
        reader_score: Tensor = None,
        targets: Tensor = None,
        proposal_score: Optional[Tensor] = None,
        proposal_log_weight: Optional[Tensor] = None,
        retriever_agg_score: Optional[Tensor] = None,
        retriever_log_p_dloc: Optional[Tensor] = None,
        reader_log_p_dloc: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Compute the loss and diagnostics. This methods assumes using three probabilities
        distributions:
            1. reader: `p_\theta(a|d, q) \propto g_\theta(q,d)`
            3. retriever: `p_\theta(d|q) \propto f_\theta(d,q)`
            3. proposal: `r_\phi(d|q) \propto f_\phi(d,q)`
            4. priority weights: `s(d) = p_\psi(d|q) / q(d)`

        Parameters
        ----------
        retriever_score
            f_\phi(d,q): shape [bs, n_opts, n_docs]
        reader_score
            f_\theta(d,q): shape [bs, n_opts, n_docs]
        targets
            a_\star: shape [bs]
        proposal_score
            f_\psi(d,q): shape [bs, n_opts, n_docs]
        proposal_log_weight
            log s(d,q): shape [bs, n_opts, n_docs]

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary of diagnostics including the loss.

        """
        alpha = format_parameter(kwargs.get("alpha", 0.0))
        reader_kl_weight = format_parameter(kwargs.get("reader_kl_weight", None))
        retriever_kl_weight = format_parameter(kwargs.get("retriever_kl_weight", None))
        eval_alpha = format_parameter(kwargs.get("eval_alpha", None))

        # evaluation temperature
        if not self.training and eval_alpha is not None:
            alpha = eval_alpha

        # rename input variables
        beta = 1 - alpha
        g_theta_ = reader_score
        f_theta_ = retriever_score
        f_phi_ = proposal_score
        log_s_ = proposal_log_weight
        targets = targets.long()

        # run diagnostics
        diagnostics = retriever_diagnostics(
            retriever_score=retriever_score,
            proposal_score=proposal_score,
            reader_score=reader_score,
            retriever_agg_score=retriever_agg_score,
            retriever_log_p_dloc=retriever_log_p_dloc,
            **kwargs,
        )

        # check inputs
        assert g_theta_ is not None
        assert f_theta_ is not None
        assert f_phi_ is not None
        assert log_s_ is not None
        assert targets is not None

        # count and replace NaNs in the scores
        try:
            nans_info = {
                f"nans/{key}": cleanup_nans(score, key)
                for key, score in [
                    ("f_theta", f_theta_),
                    ("g_theta", g_theta_),
                    ("f_phi", f_phi_),
                    ("log_s", log_s_),
                ]
            }
        except Exception as e:
            logger.error(e)
            nans_info = {}

        # compute `\zeta = \exp f_\theta - f_\phi`
        log_zeta_ = f_theta_ - f_phi_

        # normalize the importance weights
        log_s_ = log_s_.log_softmax(dim=-1)

        # compute the dfferentiable estimate of `\log p(d | q)`
        w = (log_s_ + log_zeta_).softmax(dim=-1)
        diff_log_p_D__Q_ = f_theta_ - (w.detach() * f_theta_).sum(dim=-1, keepdims=True)

        # compute the cartesian product of all variables
        args = f_theta_, log_zeta_, g_theta_, log_s_, diff_log_p_D__Q_
        f_theta, log_zeta, g_theta, log_s, diff_log_p_D__Q = batch_cartesian_product(
            *args, max_size=self.cartesian_max_size
        )

        # expand the targets
        targets_expanded = targets.view(targets.size(0), 1, 1)
        targets_expanded = targets_expanded.expand(targets.size(0), 1, log_zeta.size(2))

        # compute `\log p(A | D, Q)` and slice the `\log p(a | D, Q)`
        log_p_A__D_Q = g_theta.log_softmax(dim=1)

        log_p_a__D_Q = log_p_A__D_Q.gather(dim=1, index=targets_expanded).squeeze(1)

        # compute `s(D)`, `\zeta(D)`
        log_S = log_s.sum(1)
        log_Zeta = log_zeta.sum(1)

        # compute the importance weight `w(d,q) = p(a,d,|q) / q(d|q)` for all `A` and in `a`
        log_v = log_Zeta - (log_S + log_Zeta).logsumexp(dim=-1, keepdims=True)
        log_w_A = log_p_A__D_Q + log_v[:, None, :]
        log_w_a = log_w_A.gather(1, index=targets_expanded).squeeze(1)
        self.ess_diagnostics(diagnostics, log_w_a)
        if alpha != 0:
            self.ess_diagnostics(diagnostics, beta * log_w_a, key="ess-alpha")

        # compute the Renyi bound for alpha=0 in `A` and in `a`
        log_w_A_zero = (log_S + log_Zeta).log_softmax(dim=-1)
        L_A_zero = (log_w_A_zero[:, None, :] + log_p_A__D_Q).logsumexp(dim=-1)
        L_a_zero = L_A_zero.gather(1, index=targets[:, None]).squeeze(1)

        # compute the Renyi bound for alpha=1 in `A` and in `a`
        L_A_one = (log_S[:, None, :].exp() + log_p_A__D_Q).sum(dim=-1)
        L_a_one = L_A_one.gather(1, index=targets[:, None]).squeeze(1)

        # compute the Renyi bound in `A` and in `a`
        if alpha == 0:
            L_A_alpha = L_A_zero
            # L_a_alpha = L_a_zero
        elif alpha == 1:
            L_A_alpha = L_A_one
            # L_a_alpha = L_a_one
        else:
            # do a linear interpolation for `Beta < eps` to guarantee numerical stability
            eps = 1e-2
            beta_ = max(eps, beta)
            L_A_alpha = (1.0 / beta_) * (log_S[:, None, :] + beta_ * log_w_A).logsumexp(dim=-1)
            if beta < eps:
                u = (eps - beta) / eps
                L_A_alpha = (1 - u) * L_A_alpha + u * L_A_zero

            # L_a_alpha = L_A_alpha.gather(1, index=targets[:, None]).squeeze(1)

        # compute the gradient
        log_w = log_S + beta * (log_Zeta + log_p_a__D_Q)
        # log_w = beta * (log_S + log_Zeta + log_p_a__D_Q)
        w = log_w.softmax(dim=-1)
        log_p_a_D__q = log_p_a__D_Q + diff_log_p_D__Q.sum(1)
        grad_L_a_alpha = (w.detach() * log_p_a_D__q).sum(dim=-1)

        # compute the loss
        loss = -1 * grad_L_a_alpha

        # compute the terms for regularization and diagnostics
        kl_reader = kl_divergence(L_A_alpha, dim=1)
        diagnostics["reader/kl_uniform"] = kl_reader.mean()
        kl_retriever = kl_divergence(f_phi_, dim=2).sum(1)
        diagnostics["retriever/kl_uniform"] = kl_retriever.mean()

        if reader_kl_weight is not None:
            loss = loss + reader_kl_weight * kl_reader
        if retriever_kl_weight is not None:
            loss = loss + retriever_kl_weight * kl_retriever

        # add the relevance targets for the retriever
        diagnostics.update(
            self._get_relevance_metrics(retriever_score, kwargs.get("match_score", None))
        )

        LA_normed = L_A_alpha.log_softmax(dim=1)
        La_normed = LA_normed.gather(1, index=targets[:, None]).squeeze(1)
        H_zero = entropy(L_A_zero, dim=1).mean()
        H_alpha = entropy(LA_normed, dim=1).mean()

        # measure agreement
        pred_zero = L_A_zero.argmax(dim=1)
        pred_alpha = L_A_alpha.argmax(dim=1)
        agreement_alpha = (pred_zero == pred_alpha).float().mean()
        accuracy_alpha = (targets == pred_alpha).float().mean()

        # In-batch approx. bound
        L_inbatch = (f_theta.log_softmax(dim=-1) + g_theta.log_softmax(dim=1)).logsumexp(dim=-1)
        pred_inbatch = L_inbatch.argmax(dim=1)
        agreement_inbatch = (pred_zero == pred_inbatch).float().mean()
        accuracy_inbatch = (targets == pred_inbatch).float().mean()

        output = {
            "loss": loss,
            "reader/entropy": H_zero,
            "reader/logp": L_a_zero.detach(),
            "reader/kl_p_q": (L_a_zero - L_a_one).detach(),
            "_reader_logits_": L_A_zero.detach(),  # <- use L_alpha for the accuracy
            "_reader_scores_": reader_score.detach(),
            "_reader_targets_": targets.detach(),
            "_retriever_scores_": retriever_score.detach(),
            "_retriever_reading_logits_": retriever_score.sum(-1).detach(),
            # alpha diagnostics
            "reader/Accuracy-alpha": accuracy_alpha,
            "reader/entropy-alpha": H_alpha,
            "reader/logp-alpha": La_normed.detach(),
            "reader/agree_alpha": agreement_alpha.detach(),
            # in-batch diagnostics
            "reader/agree-inbatch": agreement_inbatch,
            "reader/Accuracy-inbatch": accuracy_inbatch,
            **diagnostics,
            **nans_info,
        }

        return output


def entropy(logp, dim=1):
    return -(logp.exp() * logp).sum(dim=dim)
