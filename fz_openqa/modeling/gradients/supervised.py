import rich
import torch
from loguru import logger
from torch import Tensor
from torch.nn import functional as F

from fz_openqa.modeling.gradients.base import Gradients
from fz_openqa.modeling.gradients.retriever_diagnostics import retriever_diagnostics
from fz_openqa.utils.pretty import pprint_batch


class SupervisedGradients(Gradients):
    def __call__(self, *, retriever_score: Tensor, match_score: Tensor, doc_ids: Tensor, **kwargs):

        if (match_score[..., :, 0] == 0).any():
            raise ValueError("First documents must have relevance > 0")
        if (match_score[..., :, 1:] > 0).any():
            logger.warning("Non-first documents must have relevance == 0")

        # run diagnostics
        diagnostics = retriever_diagnostics(
            retriever_score=retriever_score, match_score=match_score, doc_ids=doc_ids, **kwargs
        )

        # compute the loss
        retriever_targets = torch.zeros_like(retriever_score[..., 0], dtype=torch.long)
        retriever_logits = retriever_score.log_softmax(dim=-1)
        loss = -retriever_logits.gather(dim=-1, index=retriever_targets.unsqueeze(-1))

        diagnostics["loss"] = loss
        diagnostics["_retriever_targets_"] = retriever_targets.detach()
        diagnostics["_retriever_logits_"] = retriever_logits.detach()
        return diagnostics
