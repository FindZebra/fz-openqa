from typing import Optional

import torch
from torch import Tensor

from fz_openqa.modeling.gradients.retriever_diagnostics import retriever_diagnostics
from fz_openqa.modeling.gradients.supervised import SupervisedGradients
from fz_openqa.modeling.gradients.utils import kl_divergence
from fz_openqa.utils.functional import batch_reduce


class DistillationGradients(SupervisedGradients):
    def __call__(
        self,
        *,
        retriever_score: Tensor,
        proposal_score: Tensor,
        match_score: Optional[Tensor] = None,
        share_documents_across_batch: bool = False,
        **kwargs,
    ):

        if share_documents_across_batch:
            raise NotImplementedError(
                "DistillationGradients does not support "
                "sharing documents across batch `proposal_score` "
                "would be unknown if shared"
            )

        # run diagnostics
        diagnostics = retriever_diagnostics(
            retriever_score=retriever_score,
            proposal_score=proposal_score,
            match_score=match_score,
            share_documents_across_batch=share_documents_across_batch,
            **kwargs,
        )

        # loss = kl(retriever || checkpoint)
        kl = kl_divergence(retriever_score, proposal_score)
        loss = batch_reduce(kl, op=torch.mean)

        # add the relevance targets for the retriever
        diagnostics.update(self._get_relevance_metrics(retriever_score, match_score))
        diagnostics["loss"] = loss
        return diagnostics
