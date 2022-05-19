from typing import Optional

import rich
import torch
from loguru import logger
from torch import Tensor

from fz_openqa.modeling.gradients.retriever_diagnostics import retriever_diagnostics
from fz_openqa.modeling.gradients.supervised import SupervisedGradients
from fz_openqa.modeling.gradients.utils import kl_divergence
from fz_openqa.utils.functional import batch_reduce


class ContrastiveGradients(SupervisedGradients):
    def __init__(self, *args, agg: str = "prob", **kwargs):
        super().__init__(*args, **kwargs)
        self.agg = agg

    def __call__(
        self,
        *,
        targets: Tensor,
        retriever_score: Tensor,
        proposal_log_weight: Optional[Tensor] = None,
        doc_ids: Tensor = None,
        share_documents_across_batch: bool = False,
        match_score: Tensor = None,
        raw_doc_ids: Tensor = None,
        **kwargs,
    ):
        if self.agg == "prob" and share_documents_across_batch:
            raise ValueError(
                f"Sharing documents across batch is incompatible with {self.agg} "
                f"because the proposal weights would not be defined."
            )

        if share_documents_across_batch and raw_doc_ids is None:
            raise ValueError(
                "`raw_doc_ids` must be provided when setting `share_documents_across_batch=True`"
            )

        # parameters
        reader_kl_weight = kwargs.get("reader_kl_weight", None)
        retriever_kl_weight = kwargs.get("retriever_kl_weight", None)

        # run diagnostics
        diagnostics = retriever_diagnostics(
            retriever_score=retriever_score,
            match_score=match_score,
            doc_ids=doc_ids,
            raw_doc_ids=raw_doc_ids,
            proposal_log_weight=proposal_log_weight,
            share_documents_across_batch=share_documents_across_batch,
            **kwargs,
        )

        # possibly expand the logits and match_score, so they have the same shape
        # across multiple devices (DataParallel)
        retriever_score = self._expand_flattened_features(
            features=retriever_score,
            raw_doc_ids=raw_doc_ids,
            share_documents_across_batch=share_documents_across_batch,
        )
        if match_score is not None:
            match_score = self._expand_flattened_features(
                features=match_score,
                raw_doc_ids=raw_doc_ids,
                share_documents_across_batch=share_documents_across_batch,
                pad_value=0,
            )

        # clip the retriever score
        TORCH_MAX_FLOAT = {torch.float32: 1e9, torch.float16: 1e5}[retriever_score.dtype]
        # is_neg_inf = (retriever_score.isinf()) & (retriever_score < 0)
        # retriever_score = retriever_score.clamp(min=-TORCH_MAX_FLOAT)

        if self.agg == "add":
            # additive model
            M = retriever_score.view(retriever_score.size(0), -1).max(dim=1).values
            retriever_score_normalized = retriever_score - M[:, None, None]
            reader_score_ = retriever_score_normalized.logsumexp(dim=-1)
            normalizer = retriever_score_normalized.logsumexp(dim=(1, 2))
            # likelihood of the reader model
            log_p_a = reader_score_ - normalizer.unsqueeze(dim=1)
            log_p_ast = log_p_a.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
            # loss
            loss = -log_p_ast
            loss = batch_reduce(loss, op=torch.mean)
        elif self.agg == "mul":
            # multiplicative model
            retriever_score_ = retriever_score.masked_fill(torch.isinf(retriever_score), 0)
            reader_score = retriever_score_.sum(dim=-1)
            log_p_a = reader_score.log_softmax(dim=-1)
            log_p_ast = log_p_a.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
            # loss
            loss = -log_p_ast
            loss = batch_reduce(loss, op=torch.mean)
        elif self.agg == "prob":
            # Probabilistic model (additive)
            log_s = proposal_log_weight  # importance sampling log-weights
            # compute the likelihood
            log_w = (log_s + retriever_score).detach()
            log_w_flat = log_w.view(log_w.size(0), -1)
            log_denum = log_w_flat.logsumexp(dim=-1).unsqueeze(dim=1)
            log_num = log_w.logsumexp(dim=-1)
            log_p_a = log_num - log_denum
            log_p_ast = log_p_a.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

            # compute the gradients
            log_w_num = log_w.log_softmax(dim=-1)
            log_w_denum = log_w_flat.log_softmax(dim=-1)
            A = (log_w_num.detach().exp() * retriever_score).sum(dim=-1)
            A_ast = A.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
            B = (
                log_w_denum.detach().exp() * retriever_score.view(retriever_score.size(0), -1)
            ).sum(dim=-1)
            loss = -(A_ast - B)
        else:
            raise ValueError(f"Unknown aggregation: {self.agg}")

        if torch.isnan(log_p_ast).any():
            logger.warning("NaN in log_p_ast")

        # regularization
        kl_reader = batch_reduce(kl_divergence(log_p_a, dim=1), op=torch.mean)
        diagnostics["reader/kl_uniform"] = kl_reader
        retriever_score_ = retriever_score.clamp(min=-TORCH_MAX_FLOAT)
        kl_retriever = batch_reduce(kl_divergence(retriever_score_, dim=-1), op=torch.mean)
        diagnostics["retriever/kl_uniform"] = kl_retriever

        # auxiliary loss terms
        if reader_kl_weight is not None:
            loss = loss + reader_kl_weight * kl_reader
        if retriever_kl_weight is not None:
            loss = loss + retriever_kl_weight * kl_retriever

        # add the relevance targets for the retriever
        diagnostics.update(self._get_relevance_metrics(retriever_score, match_score))

        # diagnostics
        diagnostics.update(
            {
                "loss": loss,
                "reader/entropy": -(log_p_a.exp() * log_p_a).sum(dim=1).mean().detach(),
                "reader/logp": log_p_ast.detach(),
                "_reader_logits_": log_p_a.detach(),
                "_reader_scores_": log_p_a.detach(),
                "_reader_targets_": targets.detach(),
                "_retriever_scores_": retriever_score.detach(),
            }
        )
        return diagnostics
