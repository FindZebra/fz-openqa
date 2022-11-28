from pydantic import Field
from warp_pipes import pprint_batch

from fz_openqa.datamodules.pipes import torch
from fz_openqa.modeling.gradients.base import Gradients
from fz_openqa.modeling.gradients.base import GradientsOutput
from fz_openqa.modeling.gradients.base import GradientsStepOutput


class InBatchGradientsOutput(GradientsOutput):
    """Output of the InBatchGradients class."""

    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    log_p_a: torch.Tensor = Field(
        ..., alias="reader/logp", description="Log probability of the reader model log p(a)"
    )

    def dict(
        self,
        *,
        by_alias: bool = True,
        **kwargs,
    ):
        return super().dict(by_alias=by_alias, **kwargs)


class InBatchGradients(Gradients):
    def _step_end(self, step_output: GradientsStepOutput) -> InBatchGradientsOutput:
        log_p_d = step_output.f_theta.log_softmax(dim=-1)
        log_p_a = (step_output.log_p_a__d + log_p_d).logsumexp(dim=-1)
        return InBatchGradientsOutput(
            loss=-log_p_a.view(log_p_a.shape[0], -1).sum(dim=-1),
            log_p_a=log_p_a,
        )
