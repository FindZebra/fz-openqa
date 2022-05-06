import math

import torch
from loguru import logger
from torch import Tensor

from fz_openqa.modeling.gradients.base import Gradients
from fz_openqa.modeling.gradients.retriever_diagnostics import retriever_diagnostics
from fz_openqa.utils.functional import batch_reduce


class SupervisedGradients(Gradients):
    def __call__(
        self,
        *,
        retriever_score: Tensor,
        match_score: Tensor,
        doc_ids: Tensor,
        share_documents_across_batch: bool,
        raw_doc_ids: Tensor = None,
        **kwargs,
    ):

        if share_documents_across_batch and raw_doc_ids is None:
            raise ValueError(
                "`raw_doc_ids` must be provided when setting `share_documents_across_batch=True`"
            )

        # run diagnostics
        diagnostics = retriever_diagnostics(
            retriever_score=retriever_score,
            match_score=match_score,
            doc_ids=doc_ids,
            raw_doc_ids=raw_doc_ids,
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
        if retriever_score.shape != match_score.shape:
            raise ValueError(
                f"`retriever_score` and `match_score` must have the same shape. "
                f"Found: {retriever_score.shape} != {match_score.shape}."
            )

        # infer the targets
        retriever_targets = self._get_retriever_targets(match_score=match_score)

        retriever_logits = retriever_score.log_softmax(dim=-1)
        loss = -retriever_logits.gather(dim=-1, index=retriever_targets.unsqueeze(-1))
        loss = batch_reduce(loss, op=torch.mean)

        # add the relevance targets for the retriever
        diagnostics.update(self._get_relevance_metrics(retriever_score, match_score))

        diagnostics["loss"] = loss
        diagnostics["_retriever_targets_"] = retriever_targets.view(-1).detach()
        return diagnostics

    @staticmethod
    def _get_retriever_targets(*, match_score, warn: bool = True):
        # infer the binary relevance tensor from the `match_score` tensor
        relevance = (match_score.float() > 0).float()

        # check `match_score`: there is one and only one positive document,
        # and this document is placed at index 0
        if warn and (relevance.sum(-1) == 0).any():
            n = (relevance.sum(-1) == 0).float().sum()
            k = relevance[..., 0].numel()
            logger.warning(
                f"Some questions are not paired with `positive` "
                f"documents (n={n}/{k}, p={n/k:.2%})"
            )
        if warn and (relevance.sum(-1) > 1).any():
            n = (relevance.sum(-1) > 1).float().sum()
            k = relevance[..., 0].numel()
            logger.warning(
                f"Some questions are paired with more than one `positive` "
                f"documents (n={n}/{k}, p={n/k:.2%})"
            )

        # infer the retriever targets from the relevance tensor
        retriever_targets = relevance.argmax(dim=-1)

        return retriever_targets

    @staticmethod
    def _expand_flattened_features(
        *, features, raw_doc_ids, share_documents_across_batch, pad_value=-torch.inf
    ):
        if share_documents_across_batch:
            # pad the `retriever_score` to the max. possible values
            # (total number of docs if they were all unique)
            # this is required for `DataParallel`, to reduce the `_retriever_logits_`
            # and ensure they have the same dimension on all devices
            pad_dim = math.prod(raw_doc_ids.shape)
            if features.shape[-1] < pad_dim:
                n_pad = pad_dim - features.shape[-1]
                pad = pad_value + torch.zeros_like(features[..., :1]).expand(
                    *features.shape[:-1], n_pad
                )
                features = torch.cat([features, pad], dim=-1)

        return features
