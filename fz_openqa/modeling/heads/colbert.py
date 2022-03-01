from typing import Dict
from typing import Optional

import rich
import torch
import torch.nn.functional as F
from torch import einsum
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from fz_openqa.modeling.heads.dpr import DprHead
from fz_openqa.modeling.modules.utils.utils import gen_preceding_mask


class ColbertHead(DprHead):
    """Score question and document representations."""

    def __init__(
        self,
        *,
        use_mask: bool = False,
        use_answer_mask=False,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_mask = use_mask
        self.use_answer_mask = use_answer_mask
        if self.use_answer_mask:
            assert tokenizer is not None, "tokenizer must be provided if use_answer_mask is True"
            self.sep_token_id = tokenizer.sep_token_id
        else:
            self.sep_token_id = None

    def score(
        self,
        *,
        hq: Tensor,
        hd: Tensor,
        doc_ids: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        **kwargs,
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
        self,
        last_hidden_state: Tensor,
        head: str,
        mask: Optional[Tensor] = None,
        batch: Optional[Dict[str, Tensor]] = None,
        question_mask: float = 1.0,
        **kwargs,
    ) -> Tensor:

        if self.output_size is not None:
            head_op = {"document": self.d_head, "question": self.q_head}[head]
            last_hidden_state = head_op(last_hidden_state)

        if self.normalize:
            last_hidden_state = F.normalize(last_hidden_state, p=2, dim=-1)

        if self.use_mask and mask is not None:
            last_hidden_state = last_hidden_state * mask.unsqueeze(-1)
            last_hidden_state = last_hidden_state / mask.sum(1, keepdim=True)

        if self.use_answer_mask and head == "question" and question_mask < 1:
            # mask the tokens up to the first SEP token (use to separate answers from questions)
            qids = batch[f"{head}.input_ids"]
            mask = gen_preceding_mask(qids, self.sep_token_id)
            mask = torch.where(
                mask,
                torch.ones_like(mask, dtype=torch.float),
                question_mask * torch.ones_like(mask, dtype=torch.float),
            )
            last_hidden_state = last_hidden_state * mask.unsqueeze(-1)

        return last_hidden_state
