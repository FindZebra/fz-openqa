import math
from typing import Optional

import einops
import torch
from torch import Tensor

from fz_openqa.modeling.heads.base import Head


class CrossAttentionHead(Head):
    def __init__(
        self,
        input_size: int = 768,
        output_size: int = 256,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(input_size=input_size, output_size=output_size, **kwargs)
        self.num_attention_heads = num_attention_heads
        self.kv = torch.nn.Linear(input_size, 2 * output_size)
        self.q = torch.nn.Linear(input_size, output_size)
        self.residual = torch.nn.Linear(input_size, output_size)
        self.attention_dropout = torch.nn.Dropout(attention_dropout)
        self.hidden_dropout = torch.nn.Dropout(hidden_dropout)
        self.norm = torch.nn.LayerNorm(output_size)
        self.projection = torch.nn.Linear(output_size, 3 * output_size)

    def forward(
        self, *, hd: Tensor, hq: Tensor, q_mask: Optional[Tensor] = None, **kwargs
    ) -> Tensor:

        # compute the features for the attention layer
        q = self.q(hq)
        q = einops.rearrange(
            q, "bs opts l (heads h) -> bs opts heads l h", heads=self.num_attention_heads
        )
        kv = self.kv(hd)
        kv = einops.rearrange(
            kv,
            "bs opts docs l (heads h) -> bs opts docs heads l h",
            heads=self.num_attention_heads,
        )
        k, v = kv.chunk(2, dim=-1)

        # compute the cross-attention weights
        weights = torch.einsum("bohux, bodhvx -> bodhuv", q, k)
        weights = weights / math.sqrt(weights.shape[-1])
        weights = weights.softmax(dim=-1)
        weights = self.attention_dropout(weights)

        # cross-attention out
        h_qd = torch.einsum("bodhuv, bodhvx -> bodhux", weights, v)
        h_qd = einops.rearrange(
            h_qd,
            "bs opts docs heads l h -> bs opts docs l (heads h)",
            heads=self.num_attention_heads,
        )

        # apply dropout + layer norm + residual
        h_qd = self.hidden_dropout(h_qd)
        skip = self.residual(hq).unsqueeze(2)
        h_qd = self.norm(h_qd + skip)

        # pprint_batch({"h_qd": h_qd}, "Soft reader scores : 2")

        # final self-attention layer
        qkv = self.projection(h_qd)
        qkv = einops.rearrange(
            qkv,
            "bs opts docs l (heads h) -> bs opts docs heads l h",
            heads=self.num_attention_heads,
        )
        q, k, v = qkv.chunk(3, dim=-1)
        q = q[..., 0, :]  # select the CLS token of the query

        # compute the self-attention weights
        weights = torch.einsum("bodhx, bodhvx -> bodhv", q, k)
        weights = weights / math.sqrt(weights.shape[-1])
        if q_mask is not None:
            mask_ = q_mask.bool()
            mask_ = mask_.view(*mask_.shape[:2], 1, 1, mask_.shape[-1])
            weights = weights.masked_fill(~mask_, -float("inf"))
        weights = weights.softmax(dim=-1)
        weights = self.attention_dropout(weights)

        # compute the final score of shape [bs, n_opts, n_docs]
        score = torch.einsum("bodhv, bodhvx -> bod", weights, v)
        return score

    def preprocess(
        self, last_hidden_state: Tensor, head: str, mask: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement preprocess")
