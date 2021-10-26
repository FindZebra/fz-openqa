from typing import Optional

from torch import nn
from torch import Tensor
from torch.nn import functional as F
from transformers import BertPreTrainedModel

from fz_openqa.modeling.backbone import Backbone


class BertLinearHeadCls(Backbone):
    """A bert model with a linear layer to project the representation at the CLS token."""

    def __init__(
        self,
        *,
        bert: BertPreTrainedModel,
        output_size: Optional[int],
        normalize: bool = False
    ):
        super().__init__(bert=bert)
        self.bert = bert
        self.normalize = normalize
        if output_size is not None:
            self.head = nn.Linear(bert.config.hidden_size, output_size)
        else:
            self.head = None

    def forward(
        self, input_ids: Tensor, *, attention_mask: Tensor, **kwargs
    ) -> Tensor:
        last_hidden_state = self.bert(
            input_ids, attention_mask
        ).last_hidden_state
        cls_repr = last_hidden_state[:, 0]  # CLS token

        if self.head is not None:
            cls_repr = self.head(cls_repr)

        if self.normalize:
            cls_repr = F.normalize(cls_repr, p=2, dim=-1)

        return cls_repr
