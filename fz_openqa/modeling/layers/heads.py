import torch
from torch import nn
from torch import Tensor
from transformers import BertPreTrainedModel


class cls_head(nn.Module):
    """A linear head consuming the representation at the CLS token"""

    def __init__(self, bert: BertPreTrainedModel, output_size: int):
        super().__init__()
        self.linear = nn.Linear(bert.config.hidden_size, output_size)

    def forward(self, last_hidden_state: Tensor):
        cls_ = last_hidden_state[:, 0]  # CLS token
        return torch.nn.functional.normalize(self.linear(cls_), p=2, dim=-1)
