import torch.nn.functional as F
from torch import Tensor

from fz_openqa.modeling.heads import ClsHead


class ColbertHead(ClsHead):
    id: str = "colbert"

    def forward(self, last_hidden_state: Tensor, **kwargs) -> Tensor:
        context_repr = last_hidden_state

        if self.head is not None:
            context_repr = self.head(context_repr)

        if self.normalize:
            context_repr = F.normalize(context_repr, p=2, dim=-1)

        return context_repr
