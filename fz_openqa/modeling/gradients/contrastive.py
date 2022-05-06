import rich
import torch
from loguru import logger
from torch import Tensor

from fz_openqa.modeling.gradients.retriever_diagnostics import retriever_diagnostics
from fz_openqa.modeling.gradients.supervised import SupervisedGradients
from fz_openqa.modeling.gradients.utils import kl_divergence
from fz_openqa.utils.functional import batch_reduce


class ContrastiveGradients(SupervisedGradients):
    def __call__(
        self,
        *,
        targets: Tensor,
        retriever_score: Tensor,
        doc_ids: Tensor = None,
        share_documents_across_batch: bool = False,
        match_score: Tensor = None,
        raw_doc_ids: Tensor = None,
        **kwargs
    ):
        if share_documents_across_batch and raw_doc_ids is None:
            raise ValueError(
                "`raw_doc_ids` must be provided when setting `share_documents_across_batch=True`"
            )

        # parameters
        gamma = kwargs.get("gamma", 1.0)
        reader_kl_weight = kwargs.get("reader_kl_weight", None)
        retriever_kl_weight = kwargs.get("retriever_kl_weight", None)

        # run diagnostics
        diagnostics = retriever_diagnostics(
            retriever_score=retriever_score,
            match_score=match_score,
            doc_ids=doc_ids,
            raw_doc_ids=raw_doc_ids,
            share_documents_across_batch=share_documents_across_batch,
            **kwargs
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

        # reader model
        # replace padding (-inf) with zeros
        retriever_score = retriever_score.masked_fill(retriever_score < -1e-4, 0)
        retriever_score = gamma * retriever_score
        reader_score = retriever_score.sum(dim=-1)
        log_p_a = reader_score.log_softmax(dim=-1)
        log_p_ast = log_p_a.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        # regularization
        kl_reader = batch_reduce(kl_divergence(log_p_a, dim=1), op=torch.mean)
        diagnostics["reader/kl_uniform"] = kl_reader
        kl_retriever = batch_reduce(kl_divergence(retriever_score, dim=-1), op=torch.mean)
        diagnostics["retriever/kl_uniform"] = kl_retriever

        # loss
        loss = -log_p_ast
        loss = batch_reduce(loss, op=torch.mean)

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
                "_reader_scores_": reader_score.detach(),
                "_reader_targets_": targets.detach(),
                "_retriever_scores_": retriever_score.detach(),
            }
        )
        return diagnostics
