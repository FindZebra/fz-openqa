from typing import Optional

import torch.nn.functional as F
from torch import einsum
from torch import Tensor

from fz_openqa.modeling.heads.dpr import DprHead


class ColbertHead(DprHead):
    """Score question and document representations."""

    def __init__(self, *, use_mask: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_mask = use_mask

    def score(
        self,
        *,
        hq: Tensor,
        hd: Tensor,
        doc_ids: Optional[Tensor] = None,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        if not self.across_batch:
            scores = einsum("bouh, bodvh -> boduv", hq, hd)
            max_scores, _ = scores.max(-1)
            return max_scores.sum(-1)
        else:
            hd = self._flatten_documents(hd, doc_ids)
            scores = einsum("bouh, mvh -> bomuv", hq, hd)
            max_scores, _ = scores.max(-1)
            return max_scores.sum(-1)

    def preprocess(
        self, last_hidden_state: Tensor, head: str, mask: Optional[Tensor] = None
    ) -> Tensor:

        if self.output_size is not None:
            head = {"document": self.d_head, "question": self.q_head}[head]
            last_hidden_state = head(last_hidden_state)

        if self.normalize:
            last_hidden_state = F.normalize(last_hidden_state, p=2, dim=-1)
            # cls_repr /= float(cls_repr.shape[-1])**0.5

        if self.use_mask and mask is not None:
            last_hidden_state = last_hidden_state * mask.unsqueeze(-1)

        return last_hidden_state
