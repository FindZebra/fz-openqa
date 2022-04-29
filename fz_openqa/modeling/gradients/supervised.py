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

        # infer the targets
        retriever_targets = self._get_retriever_targets(
            doc_ids=doc_ids,
            raw_doc_ids=raw_doc_ids,
            match_score=match_score,
            share_documents_across_batch=share_documents_across_batch,
        )

        # possibly expand the logits, so they have the same shape
        # across multiple devices in DataParallel
        retriever_score = self._format_retriever_logits(
            retriever_score=retriever_score,
            raw_doc_ids=raw_doc_ids,
            share_documents_across_batch=share_documents_across_batch,
        )

        retriever_logits = retriever_score.log_softmax(dim=-1)
        loss = -retriever_logits.gather(dim=-1, index=retriever_targets.unsqueeze(-1))
        loss = batch_reduce(loss, op=torch.mean)

        diagnostics["loss"] = loss
        diagnostics["_retriever_targets_"] = retriever_targets.view(-1).detach()
        diagnostics["_retriever_logits_"] = (
            retriever_logits.view(-1, retriever_logits.size(-1)).detach(),
        )
        return diagnostics

    @staticmethod
    def _get_retriever_targets(
        *, match_score, doc_ids, raw_doc_ids, share_documents_across_batch, warn: bool = True
    ):

        if raw_doc_ids is not None and not raw_doc_ids.shape == match_score.shape:
            # the relevance tensor must always be provided in its non-flattened form
            raise ValueError(
                "`match_score` and `raw_doc_ids` must be provided in their"
                "non-flattened form and therefore must both have the same shape. "
                f"Found: match_score: {match_score.shape}, raw_doc_ids: {raw_doc_ids}"
            )

        # infer the binary relevance tensor from the `match_score` tensor
        relevance = (match_score > 0).float()

        # check `match_score`: there is one and only one positive document,
        # and this document is placed at index 0
        if warn and (relevance.sum(-1) == 0).any():
            n = (relevance.sum(-1) == 0).float().sum()
            k = relevance[..., 0].numel()
            logger.warning(
                f"Some questions are not paired with `positive` "
                f"documents (n={n}/{k}, p={n/k:.2%})"
            )
        if warn and (match_score[..., :, 1:] > 0).any():
            n = (relevance.sum(-1) > 1).float().sum()
            k = relevance[..., 0].numel()
            logger.warning(
                f"Some questions are paired with more than one `positive` "
                f"documents (n={n}/{k}, p={n/k:.2%})"
            )

        # infer the retriever targets from the relevance tensor
        retriever_targets = relevance.argmax(dim=-1)

        if share_documents_across_batch:
            # if the documents have been flattened, convert the position of the
            # targets given in the non-flat version to the flat version
            target_ids = raw_doc_ids.gather(dim=-1, index=retriever_targets.unsqueeze(-1))
            retriever_targets = (target_ids - doc_ids.unsqueeze(0)).abs().argmin(dim=-1)

        return retriever_targets

    @staticmethod
    def _format_retriever_logits(*, retriever_score, raw_doc_ids, share_documents_across_batch):
        if share_documents_across_batch:
            # pad the `retriever_score` to the max. possible values
            # (total number of docs if they were all unique)
            # this is required for `DataParallel`, to reduce the `_retriever_logits_`
            # and ensure they have the same dimension on all devices
            pad_dim = math.prod(raw_doc_ids.shape)
            if retriever_score.shape[-1] < pad_dim:
                n_pad = pad_dim - retriever_score.shape[-1]
                pad = -torch.inf + torch.zeros_like(retriever_score[..., :1]).expand(
                    *retriever_score.shape[:-1], n_pad
                )
                retriever_score = torch.cat([retriever_score, pad], dim=-1)

        return retriever_score
