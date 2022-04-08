import math
from typing import Dict
from typing import Optional

import torch
import torch.nn.functional as F
from torch import einsum
from torch import nn
from torch import Tensor

from fz_openqa.modeling.heads import DprHead
from fz_openqa.modeling.layers import BayesianLinear


class ZeroHead(DprHead):
    """Score question without documents."""

    def __init__(
        self,
        *,
        normalize: bool = False,
        bias: bool = True,
        bayesian: bool = False,
        learn_temperature: bool = False,
        temperature: float = 1.0,
        **kwargs
    ):
        super(ZeroHead, self).__init__(**kwargs)
        self.bias = bias

        Layer = nn.Linear if not bayesian else BayesianLinear

        self.normalize = normalize
        if self.output_size is not None:
            self.q_head = Layer(self.input_size, self.output_size, bias=self.bias)

        # temperate
        log_temperature = torch.tensor(math.log(temperature), dtype=torch.float)
        if learn_temperature:
            self.log_temperature = nn.Parameter(log_temperature)
        else:
            self.register_buffer("log_temperature", log_temperature)

    def forward(
        self,
        *,
        hq: Tensor,
        doc_ids: Optional[Tensor] = None,
        q_mask: Optional[Tensor] = None,
        d_mask: Optional[Tensor] = None,
        batch: Dict[str, Tensor] = None,
        **kwargs
    ) -> Tensor:

        # preprocess
        hq = self._preprocess(hq, "question", mask=q_mask, batch=batch, **kwargs)

        # compute the score
        return self.score(hq=hq, batch=batch, **kwargs)

    def score(self, *, hq: Tensor, **kwargs) -> Tensor:
        return einsum("boh -> bo", hq)

    def _preprocess(
        self, last_hidden_state: Tensor, head: str, mask: Optional[Tensor] = None, **kwargs
    ) -> Optional[Tensor]:
        cls_repr = last_hidden_state[..., 0, :]  # CLS token
        if head == "document":
            return None

        if self.output_size is not None:
            head = {"question": self.q_head}[head]
            cls_repr = head(cls_repr)

        if self.normalize:
            cls_repr = F.normalize(cls_repr, p=2, dim=-1)

        if head == "question":
            cls_repr = cls_repr / self.temperature()

        return cls_repr
