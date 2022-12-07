from __future__ import annotations

import abc
import warnings
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional

import rich
import torch
from pydantic import BaseModel
from pydantic import Field
from torch import nn


class GradientsInput(BaseModel):
    """Base class for model outputs."""

    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    log_p_a__qd: torch.Tensor = Field(
        ...,
        alias="lm.logp_a",
        description="Log reader probability `p(a | [q;d])`",
    )
    log_p_qd: torch.Tensor = Field(
        ...,
        alias="lm.logp_q",
        description="Log reader probability `p([q;d])`",
    )
    log_p_lm: torch.Tensor = Field(
        ...,
        alias="lm.logp",
        description="Log reader probability `p([a;q;d])`",
    )
    lm_mc_logits: torch.Tensor = Field(
        None,
        alias="lm.mc_logits",
        description="Logits for multiple-choice answers",
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
        None,
        alias="document.proposal_score",
        description="Score of the proposal distribution",
    )
    log_s: torch.Tensor = Field(
        None,
        alias="document.proposal_log_weight",
        description="Log importance weight of the proposal distribution",
    )


class GradientsStepOutput(BaseModel):
    """Base class for gradients step outputs."""

    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    log_p_a__qd: torch.Tensor = Field(
        ...,
        alias="lm.logp_a",
        description="Log reader probability `p(a | [q;d])`",
    )
    log_p_qd: torch.Tensor = Field(
        ...,
        alias="lm.logp_q",
        description="Log reader probability `p([q;d])`",
    )
    log_p_lm: torch.Tensor = Field(
        ...,
        alias="lm.logp",
        description="Log reader probability `p([a;q;d])`",
    )
    lm_mc_logits: Optional[torch.Tensor] = Field(
        None,
        alias="lm.mc_logits",
        description="Logits for multiple-choice answers",
    )
    f_theta: Optional[torch.Tensor] = Field(
        ...,
        alias="document.retriever_score",
        description="Score of the retriever",
    )
    f_phi: Optional[torch.Tensor] = Field(
        ...,
        alias="document.proposal_score",
        description="Score of the proposal distribution",
    )
    log_s: Optional[torch.Tensor] = Field(
        ...,
        alias="document.proposal_log_weight",
        description="Log importance weight of the proposal distribution",
    )
    retriever_entropy: Optional[torch.Tensor] = Field(
        None,
        alias="document.retriever_entropy",
        description="Entropy of the retriever",
    )


class GradientsOutput(BaseModel):
    """Base class for gradients step outputs."""

    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    loss: torch.Tensor = Field(
        ...,
        description="Final loss to be differentiated. Shape is (batch_size,).",
    )
    retriever_entropy: Optional[torch.Tensor] = Field(
        ...,
        alias="document.retriever_entropy",
        description="Entropy of the retriever",
    )


class Gradients(nn.Module):
    required_features: List[str] = []

    def __init__(self, **_):
        super().__init__()

    def __call__(self, data: GradientsInput) -> GradientsOutput:
        step_output = self.step(data)
        output = self.step_end(step_output)
        return output

    def step(self, data: Dict | GradientsInput) -> GradientsStepOutput:
        if isinstance(data, dict):
            data = GradientsInput(**data)

        # get the document and question embeddings
        hd = data.document_vector
        hq = data.question_vector

        # compute the inner product between the document and question embeddings
        if hd is None or hq is None:
            warnings.warn(
                "Document or question vector is None, "
                "setting the retriever score to zero for "
                f"`{type(self).__name__}` computation."
            )
            if data.f_phi is not None:
                retriever_score = torch.zeros_like(data.f_phi)
            else:
                retriever_score = torch.zeros_like(data.log_p_lm)
        else:
            retriever_score = torch.einsum("...h, ...dh -> ... d", hq, hd)

        # format the output
        return GradientsStepOutput(
            f_theta=retriever_score, **data.dict(), **self._get_retriever_metrics(retriever_score)
        )

    def step_end(self, step_output: Dict | GradientsStepOutput) -> GradientsOutput:
        if isinstance(step_output, dict):
            step_output = GradientsStepOutput(**step_output)
        return self._step_end(step_output)

    @abc.abstractmethod
    def _step_end(self, step_output: GradientsStepOutput) -> GradientsOutput:
        ...

    @staticmethod
    def _get_retriever_metrics(retriever_scores) -> Dict:
        if retriever_scores.ndim <= 1:
            return {}
        retriever_log_probs = torch.log_softmax(retriever_scores, dim=-1)
        entropy = -torch.sum(torch.exp(retriever_log_probs) * retriever_log_probs, dim=-1)
        return {
            "retriever_entropy": entropy.view(entropy.shape[0], -1).mean(dim=1),
        }

    @staticmethod
    @torch.no_grad()
    def ess_diagnostics(diagnostics, log_W, key="ess"):
        K = log_W.size(-1)
        log_ess = 2 * log_W.logsumexp(dim=-1) - (2 * log_W).logsumexp(dim=-1)
        diagnostics[f"{key}/mean"] = log_ess.exp().mean()
        diagnostics[f"{key}/ratio-mean"] = log_ess.exp().mean() / K
        diagnostics[f"{key}/max"] = log_W.max().exp()
