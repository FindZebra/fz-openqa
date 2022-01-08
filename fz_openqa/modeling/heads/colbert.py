from typing import Optional

import einops
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from transformers import BertPreTrainedModel

from fz_openqa.modeling.heads.base import Head


class ColbertHead(Head):
    id: str = "colbert"

    def __init__(
        self,
        *,
        bert: BertPreTrainedModel,
        output_size: Optional[int],
        kernel_size: Optional[int] = None,
        downscaling: int = 1,
        normalize: bool = False,
        **kwargs
    ):
        super(ColbertHead, self).__init__(bert=bert, output_size=output_size, **kwargs)

        self.normalize = normalize
        if output_size is None and kernel_size is None and downscaling == 1:
            self.head = None
        elif kernel_size is None:
            assert downscaling == 1
            self.head = nn.Linear(bert.config.hidden_size, output_size, bias=self.bias)
        else:
            self.head = nn.Conv1d(
                bert.config.hidden_size,
                output_size,
                kernel_size=(kernel_size,),
                stride=(downscaling,),
                padding=(kernel_size // 2,),
                bias=self.bias,
            )

    def forward(
        self, last_hidden_state: Tensor, *, mask: Optional[Tensor] = None, **kwargs
    ) -> Tensor:
        context_repr = last_hidden_state

        if self.head is not None:
            if isinstance(self.head, nn.Conv1d):
                context_repr = einops.rearrange(context_repr, "... l h -> ... h l")
                context_repr = self.head(context_repr)
                context_repr = einops.rearrange(context_repr, "... h l -> ... l h")
            else:
                context_repr = self.head(context_repr)

        if self.normalize:
            context_repr = F.normalize(context_repr, p=2, dim=-1)

        if mask is not None:
            context_repr = context_repr * mask.unsqueeze(-1)

        return context_repr
