from pydantic import Field

from fz_openqa.datamodules.pipes import torch
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

    def dict(
        self,
        *,
        by_alias: bool = True,
        **kwargs,
    ):
        return super().dict(by_alias=by_alias, **kwargs)


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

        return InBatchGradientsOutput(
            loss=loss,
            **step_output.dict(by_alias=False),
        )
