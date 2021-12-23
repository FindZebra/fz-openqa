from abc import ABC
from abc import abstractmethod

from torch import nn
from torch import Tensor
from transformers import BertPreTrainedModel


class Head(nn.Module, ABC):
    id: str = "base"

    def __init__(self, *, bert: BertPreTrainedModel, output_size: int, **kwargs):
        super(Head, self).__init__()
        self.input_size = bert.config.hidden_size
        self.output_size = output_size

    @abstractmethod
    def forward(self, last_hidden_state: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError
