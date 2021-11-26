from typing import Optional

import torch.nn.functional as F
from torch import nn
from torch import Tensor
from transformers import BertPreTrainedModel

from fz_openqa.modeling.heads import ClsHead
from fz_openqa.modeling.heads.base import Head


class ColbertHead(ClsHead):
    id: str = "colbert"

    def forward(self, last_hidden_state: Tensor, **kwargs) -> Tensor:
        context_repr = last_hidden_state

        if self.head is not None:
            context_repr = self.head(context_repr)

        if self.normalize:
            context_repr = F.normalize(context_repr, p=2, dim=-1)

        return context_repr
