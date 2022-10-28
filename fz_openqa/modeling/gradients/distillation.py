from copy import copy
from typing import Optional

import rich
import torch
from torch import Tensor

from fz_openqa.modeling.gradients.retriever_diagnostics import retriever_diagnostics
from fz_openqa.modeling.gradients.supervised import SupervisedGradients
from fz_openqa.modeling.gradients.utils import kl_divergence
from fz_openqa.utils.functional import batch_reduce


class DistillationGradients(SupervisedGradients):
    required_features = [
        "document.proposal_score",
    ]

    def __call__(
        self,
        **data,
    ):
        data = copy(data)

        # unpacking data
        retriever_score = data["retriever_score"]
        proposal_score = data["document.proposal_score"]

        # run diagnostics
        # run diagnostics
        diagnostics = retriever_diagnostics(**data, proposal_score=proposal_score)

        # loss = kl(retriever || checkpoint)
        # kl = kl_divergence(retriever_score, proposal_score)

        # match the moments of p_retriever and p_checkpoint
        masked = proposal_score.isinf()
        retriever_score_ = retriever_score.masked_fill(masked, 0)
        proposal_score_ = proposal_score.masked_fill(masked, 0)
        moment_diff = (retriever_score_ - proposal_score_) ** 2
        loss = batch_reduce(moment_diff, op=torch.mean)

        diagnostics["loss"] = loss
        return diagnostics
