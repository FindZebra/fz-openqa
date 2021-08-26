import torch
from torch import nn
from torch import Tensor
from transformers import BertPreTrainedModel


class cls_head(nn.Module):
    """A linear head consuming the representation at the CLS token"""

    def __init__(
        self,
        bert: BertPreTrainedModel,
        output_size: int,
        normalize: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(bert.config.hidden_size, output_size)
        self.normalize = normalize

    def forward(self, last_hidden_state: Tensor):
        cls_ = last_hidden_state[:, 0]  # CLS token
        h = self.linear(cls_)
        if self.normalize:
            h = torch.nn.functional.normalize(h, p=2, dim=-1)
        return h
