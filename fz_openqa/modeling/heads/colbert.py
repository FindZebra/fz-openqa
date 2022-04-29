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
from fz_openqa.utils.metric_type import MetricType


class ColbertHead(DprHead):
    """Score question and document representations."""

    def __init__(
        self,
        *,
        use_mask: bool = True,
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
            f"compute_agg_score={self.compute_agg_score}, "
            f"metric_type={self.metric_type}"
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
        tau = kwargs.get("tau", None)

        diagnostics = {}

        # compute the aggregated score
        if self.compute_agg_score:
            hd_flat, _ = self._flatten_documents(hd, doc_ids=doc_ids)
            hq_flat = hq.view(-1, *hq.shape[-2:])
            agg_retriever_score = einsum("buh, mvh -> bmuv", hq_flat, hd_flat)
            agg_retriever_score, _ = self._reduce_doc_vectors(agg_retriever_score, tau)
            agg_retriever_score = agg_retriever_score.sum(-1)
            diagnostics["agg_score"] = agg_retriever_score

        # compute the score using self-attention
        if self.use_soft_score:
            # split hq and hd
            hq_a, hq_v = hq.chunk(2, dim=-1)
            hd_a, hd_v = hd.chunk(2, dim=-1)

            # attention model
            attn_scores = einsum("bouh, bodvh -> boduv", hq_a, hd_a)
            attn_scores = attn_scores.mul(1.0 / hq_a.shape[-1])
            attn_scores = attn_scores.masked_fill(attn_scores == 0, -1e3)
            log_p_dloc = attn_scores.log_softmax(dim=-1)

            # values
            values = einsum("bouh, bodvh -> boduv", hq_v, hd_v)
            scores = (log_p_dloc.exp() * values).sum(dim=(-2, -1))

        else:
            bs = hq.shape[:-2]
            shared_batch = self._is_shared_batch_dims(hd=hd, hq=hq, bs=bs, expected_hd_dim=3)

            # reshape `hq` and `hd`
            hq = hq.view(-1, *hq.shape[len(bs) :])
            if shared_batch:
                hd = hd.view(-1, *hd.shape[len(bs) :])

            # infer the masks
            dmask_zero = hd.abs().sum(-1) == 0
            qmask_zero = hq.abs().sum(-1) == 0

            # compute the token scores
            if shared_batch:
                # compute the token-level Colbert scores
                if self.metric_type == MetricType.inner_product:
                    scores = einsum("buh, bdvh -> bduv", hq, hd)
                elif self.metric_type == MetricType.euclidean:
                    _hq = hq[:, None, :, None, :]
                    _hd = hd[:, :, None, :, :]
                    scores = -1 * (_hq - _hd).pow(2).sum(-1).pow(0.5)
                else:
                    raise ValueError(f"Unknown `metric_type`: {self.metric_type}")

                # expand the document masks
                dmask_zero = dmask_zero[:, :, None, :]
            else:
                if self.metric_type == MetricType.inner_product:
                    scores = einsum("buh, dvh -> bduv", hq, hd)
                elif self.metric_type == MetricType.euclidean:
                    _hq = hq[:, None, :, None, :]
                    _hd = hd[None, :, None, :, :]
                    scores = -1 * (_hq - _hd).pow(2).sum(-1).pow(0.5)
                else:
                    raise ValueError(f"Unknown `metric_type`: {self.metric_type}")

                # expand the document masks
                dmask_zero = dmask_zero[None, :, None, :]

            # mask the document vectors
            scores = scores.masked_fill(dmask_zero, -torch.inf)

            # sum over document tokens
            scores, log_p_dloc = self._reduce_doc_vectors(scores, tau)

            # apply the masking to the query tokens
            qmask_zero = qmask_zero[:, None, :]
            scores = scores.masked_fill(qmask_zero, 0)

            # sum over the query tokens
            scores = scores.sum(-1)

            # reshape
            scores = scores.view(*bs, scores.shape[-1])

        # aggregate document loc probs over q
        log_nq = math.log(log_p_dloc.size(-2))
        log_p_dloc = (log_p_dloc - log_nq).logsumexp(dim=-2)
        diagnostics["log_p_dloc"] = log_p_dloc

        return scores, diagnostics

    def _reduce_doc_vectors(self, scores, tau, dim=-1):
        # attention_scores
        attn_scores = scores.clone()
        log_attn_scores = attn_scores.log_softmax(dim=dim)

        # sample locations and reduce over `document` dimension
        if tau is None:
            q_scores = scores.max(dim=dim).values
        else:
            tau_ = tau if tau > 0 else 1.0
            q_locs = torch.nn.functional.gumbel_softmax(
                log_attn_scores, tau=tau_, hard=(tau <= 0), dim=dim
            )
            q_scores = (q_locs * scores).sum(dim=dim)
        return q_scores, log_attn_scores

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
            # by convention: set the vectors to `zero` for masked token.
            # Subsequent code infer vectors that are *exactly* zero as being padded
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
