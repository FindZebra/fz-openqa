import abc
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional

import torch
from torch import nn


class Gradients(nn.Module):
    required_features: List[str] = []

    def __init__(self, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, **kwargs) -> Dict:
        ...

    def step(self, **data) -> Dict:
        return self.__call__(**data)

    def step_end(self, **data) -> Dict:
        return {}

    @staticmethod
    def _get_relevance_metrics(retriever_scores, match_score: Optional[torch.Tensor]) -> Dict:
        diagnostics = {}
        if match_score is None:
            return diagnostics
        targets = (match_score > 0).view(-1, match_score.size(-1)).detach()
        retriever_scores = retriever_scores.view(-1, retriever_scores.size(-1)).detach()
        logits = retriever_scores.log_softmax(dim=-1)
        diagnostics["_retriever_binary_targets_"] = targets
        diagnostics["_retriever_logits_"] = logits
        diagnostics["retrieval/n_total"] = torch.ones_like(targets.float()).sum(-1)
        diagnostics["retrieval/n_positive"] = targets.float().sum(-1)
        return diagnostics

    @staticmethod
    @torch.no_grad()
    def ess_diagnostics(diagnostics, log_W, key="ess"):
        K = log_W.size(-1)
        log_ess = 2 * log_W.logsumexp(dim=-1) - (2 * log_W).logsumexp(dim=-1)
        diagnostics[f"{key}/mean"] = log_ess.exp().mean()
        diagnostics[f"{key}/ratio-mean"] = log_ess.exp().mean() / K
        diagnostics[f"{key}/max"] = log_W.max().exp()


class Space(Enum):
    EXP = "exp"
    LOG = "log"
