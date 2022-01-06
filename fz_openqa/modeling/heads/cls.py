from typing import Optional

import torch.nn.functional as F
from torch import nn
from torch import Tensor
from transformers import BertPreTrainedModel

from fz_openqa.modeling.heads.base import Head


class ClsHead(Head):
    id: str = "dense"

    def __init__(
        self,
        *,
        bert: BertPreTrainedModel,
        output_size: Optional[int],
        normalize: bool = False,
        **kwargs
    ):
        super(ClsHead, self).__init__(bert=bert, output_size=output_size, **kwargs)

        self.normalize = normalize
        if output_size is not None:
            self.head = nn.Linear(bert.config.hidden_size, output_size)
        else:
            self.head = None

    def forward(self, last_hidden_state: Tensor, **kwargs) -> Tensor:
        cls_repr = last_hidden_state[:, 0]  # CLS token

        if self.head is not None:
            cls_repr = self.head(cls_repr)

        if self.normalize:
            cls_repr = F.normalize(cls_repr, p=2, dim=-1)

        return cls_repr
