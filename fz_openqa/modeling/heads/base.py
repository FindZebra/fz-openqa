from torch import nn
from torch import Tensor
from transformers import BertPreTrainedModel


class Head(nn.Module):
    def __init__(self, *, bert: BertPreTrainedModel, output_size: int):
        super(Head, self).__init__()
        self.input_size = bert.config.hidden_size
        self.output_size = output_size

    def forward(self, last_hidden_state: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError
