from torch import nn
from torch import Tensor
from transformers import BertPreTrainedModel


class Backbone(nn.Module):
    """A bert model with a linear layer to project the representation at the CLS token."""

    def __init__(self, *, bert: BertPreTrainedModel):
        super(Backbone, self).__init__()
        self.bert = bert

    def forward(
        self, input_ids: Tensor, *, attention_mask: Tensor, **kwargs
    ) -> Tensor:
        raise NotImplementedError
