import torch
from pydantic import Field

from fz_openqa.modeling.datastruct import METRICS_PREFIX
from fz_openqa.modeling.gradients.base import Gradients
from fz_openqa.modeling.gradients.base import GradientsOutput
from fz_openqa.modeling.gradients.base import GradientsStepOutput


class InBatchGradientsOutput(GradientsOutput):
    """Output of the InBatchGradients class."""

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
    lm_ppl: torch.Tensor = Field(
        None,
        alias="lm.ppl",
        description="Perplexity of the language model",
    )

    lm_mc_logits: torch.Tensor = Field(
        None,
        alias=f"{METRICS_PREFIX}answer.logits",
        description="Logits for multiple-choice answers",
    )

    def dict(
        self,
        *,
        by_alias: bool = True,
        **kwargs,
    ):
        return super().dict(by_alias=by_alias, **kwargs)


class LanguageModellingGradients(Gradients):
    def _step_end(self, step_output: GradientsStepOutput) -> InBatchGradientsOutput:

        # estimate log p(a | q)
        log_p_d = step_output.f_theta.log_softmax(dim=-1)
        if step_output.log_s is not None:
            log_p_lm = (step_output.log_p_lm + step_output.log_s).logsumexp(dim=-1)
            # todo: perplexity
        else:
            assert step_output.log_p_lm.ndim == 1
            log_p_lm = step_output.log_p_lm

        # loss (negative log likelihood)
        loss = -log_p_lm.view(log_p_lm.shape[0], -1).sum(dim=-1)

        # multiple-choice logits
        if step_output.lm_mc_logits is not None:
            lm_mc_logits = (step_output.lm_mc_logits).log_softmax(dim=-1)
            if lm_mc_logits.ndim > 2:
                lm_mc_logits = (lm_mc_logits + log_p_d.unsqueeze(-1)).logsumexp(dim=1)
        else:
            lm_mc_logits = None

        # format the output
        step_output = step_output.dict(by_alias=False)
        step_output["lm_mc_logits"] = lm_mc_logits
        return InBatchGradientsOutput(
            loss=loss,
            **step_output,
        )


class InBatchGradients(Gradients):
    def __init__(self, model_context: bool = False, **kwargs):
        super().__init__()
        self.model_context = model_context

    def _step_end(self, step_output: GradientsStepOutput) -> InBatchGradientsOutput:

        # estimate log p(a | q)
        log_p_d = step_output.f_theta.log_softmax(dim=-1)
        log_p_a__q = (step_output.log_p_a__qd + log_p_d).logsumexp(dim=-1)

        # loss (negative log likelihood)
        loss = -log_p_a__q.view(log_p_a__q.shape[0], -1).sum(dim=-1)
        if self.model_context:
            log_p_qd = step_output.log_p_qd
            log_p_qd = log_p_qd.view(log_p_qd.shape[0], -1).mean(dim=-1)
            loss = loss - log_p_qd

        # multiple-choice logits
        if step_output.lm_mc_logits is not None:
            lm_mc_logits = (step_output.lm_mc_logits).log_softmax(dim=-1)
            lm_mc_logits = (lm_mc_logits + log_p_d.unsqueeze(-1)).logsumexp(dim=1)
        else:
            lm_mc_logits = None

        # format the output
        step_output = step_output.dict(by_alias=False)
        step_output["lm_mc_logits"] = lm_mc_logits
        return InBatchGradientsOutput(
            loss=loss,
            **step_output,
        )
