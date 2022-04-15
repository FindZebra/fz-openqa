import math
from typing import Dict
from typing import Optional

import rich
import torch
import torch.nn.functional as F
from loguru import logger
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
        use_soft_score: bool = False,
        compute_agg_score: bool = False,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_mask = use_mask
        self.use_answer_mask = use_answer_mask
        self.use_soft_score = use_soft_score
        self.compute_agg_score = compute_agg_score
        if self.use_answer_mask:
            assert tokenizer is not None, "tokenizer must be provided if use_answer_mask is True"
            self.sep_token_id = tokenizer.sep_token_id
        else:
            self.sep_token_id = None

        logger.info(
            f"Initialized {self.__class__.__name__} (id={self.id}) with "
            f"use_mask={self.use_mask}, "
            f"use_answer_mask={self.use_answer_mask}, "
            f"soft_score={self.use_soft_score}, "
            f"compute_agg_score={self.compute_agg_score}"
        )

    def score(
        self,
        *,
        hq: Tensor,
        hd: Tensor,
        doc_ids: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        **kwargs,
    ) -> (Tensor, Dict):

        diagnostics = {}

        # compute the aggregated score
        if self.compute_agg_score:
            hd_flat = self._flatten_documents(hd, doc_ids)
            hq_flat = hq.view(-1, *hq.shape[-2:])
            agg_retriever_score = einsum("buh, mvh -> bmuv", hq_flat, hd_flat)
            max_scores, _ = agg_retriever_score.max(-1)
            agg_retriever_score = max_scores.sum(-1)
            diagnostics["agg_retriever_score"] = agg_retriever_score

        # compute the score using self-attention
        if self.use_soft_score:
            if self.across_batch:
                raise NotImplementedError("soft_score across_batch not implemented")

            # split hq and hd
            hq_a, hq_v = hq.chunk(2, dim=-1)
            hd_a, hd_v = hd.chunk(2, dim=-1)

            # attention model
            attn_scores = einsum("bouh, bodvh -> boduv", hq_a, hd_a)
            attn_scores = attn_scores.masked_fill(attn_scores == 0, -1e9)
            attn_scores = attn_scores.softmax(dim=-1)

            # values
            values = einsum("bouh, bodvh -> boduv", hq_v, hd_v)
            scores = (attn_scores * values).sum(dim=(-2, -1))

        elif not self.across_batch:
            # compute the basic Colbert scores
            scores = einsum("bouh, bodvh -> boduv", hq, hd)
            max_scores, _ = scores.max(-1)
            scores = max_scores.sum(-1)
        else:
            # compute the Colbert scores for ALL documents within the batch
            hd = self._flatten_documents(hd, doc_ids)
            scores = einsum("bouh, mvh -> bomuv", hq, hd)
            max_scores, _ = scores.max(-1)
            scores = max_scores.sum(-1)

        return scores, diagnostics

    def _preprocess(
        self,
        last_hidden_state: Tensor,
        head: str,
        mask: Optional[Tensor] = None,
        batch: Optional[Dict[str, Tensor]] = None,
        question_mask: float = 1.0,
        weights: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        head_kwargs = {"weights": weights} if weights is not None else {}
        if self.output_size is not None:
            head_op = {"document": self.d_head, "question": self.q_head}[head]
            last_hidden_state = head_op(last_hidden_state, **head_kwargs)

        if self.normalize:
            if not self.use_soft_score:
                last_hidden_state = F.normalize(last_hidden_state, p=2, dim=-1)
            else:
                h_a, h_v = last_hidden_state.chunk(2, dim=-1)
                h_v = F.normalize(h_v, p=2, dim=-1)
                last_hidden_state = torch.cat([h_a, h_v], dim=-1)

        if self.use_mask and mask is not None:
            last_hidden_state = last_hidden_state * mask.unsqueeze(-1)
            # last_hidden_state = last_hidden_state / mask.sum(1, keepdim=True)

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

        if head == "question":
            last_hidden_state = self.scale(last_hidden_state)

        return last_hidden_state
