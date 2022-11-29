from __future__ import annotations

import abc
import warnings
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional

import torch
from pydantic import BaseModel
from pydantic import Field
from torch import nn


class GradientsInput(BaseModel):
    """Base class for model outputs."""

    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    log_p_a__d: torch.Tensor = Field(
        ...,
        alias="lm.logp",
        description="Log probability of the reader model",
    )
    document_vector: Optional[torch.Tensor] = Field(
        None,
        alias="document.pooler_output",
        description="Document vector",
    )
    question_vector: Optional[torch.Tensor] = Field(
        None,
        alias="question.pooler_output",
        description="Question vector",
    )
    f_phi: torch.Tensor = Field(
        ...,
        alias="document.proposal_score",
        description="Score of the proposal distribution",
    )
    log_s: torch.Tensor = Field(
        ...,
        alias="document.proposal_log_weight",
        description="Log importance weight of the proposal distribution",
    )


class GradientsStepOutput(BaseModel):
    """Base class for gradients step outputs."""

    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    log_p_a__d: torch.Tensor = Field(
        ...,
        alias="lm.logp",
        description="Log probability of the reader model",
    )
    f_theta: torch.Tensor = Field(
        ...,
        alias="document.retriever_score",
        description="Score of the retriever",
    )
    f_phi: torch.Tensor = Field(
        ...,
        alias="document.proposal_score",
        description="Score of the proposal distribution",
    )
    log_s: torch.Tensor = Field(
        ...,
        alias="document.proposal_log_weight",
        description="Log importance weight of the proposal distribution",
    )


class GradientsOutput(BaseModel):
    """Base class for gradients step outputs."""

    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    loss: torch.Tensor


class Gradients(nn.Module):
    required_features: List[str] = []

    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, data: GradientsInput) -> GradientsOutput:
        step_output = self.step(data)
        output = self.step_end(step_output)
        return output

    def step(self, data: Dict | GradientsInput) -> GradientsStepOutput:
        if isinstance(data, dict):
            data = GradientsInput(**data)
        # compute the document score
        hd = data.document_vector
        hq = data.question_vector
        if hd is None or hq is None:
            warnings.warn(
                "Document or question vector is None, "
                "setting the retriever score to zero for "
                f"`{type(self).__name__}` computation."
            )
            retriever_score = torch.zeros_like(data.f_phi)
        else:
            retriever_score = torch.einsum("...h, ...dh -> ... d", hq, hd)
        return GradientsStepOutput(f_theta=retriever_score, **data.dict())

    def step_end(self, step_output: Dict | GradientsStepOutput) -> GradientsOutput:
        if isinstance(step_output, dict):
            step_output = GradientsStepOutput(**step_output)
        return self._step_end(step_output)

    @abc.abstractmethod
    def _step_end(self, step_output: GradientsStepOutput) -> GradientsOutput:
        ...

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
