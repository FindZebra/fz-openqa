from typing import List
from typing import Optional

import torch
from transformers import PreTrainedTokenizerFast

from fz_openqa.datamodules.pipes import AddPrefix
from fz_openqa.datamodules.pipes import ApplyToAll
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes import Flatten
from fz_openqa.datamodules.pipes import Lambda
from fz_openqa.datamodules.pipes import Nest
from fz_openqa.datamodules.pipes import ReplaceInKeys
from fz_openqa.datamodules.pipes import Sequential


class CollateAsTensor(Sequential):
    """A pipeline to concatenate simple attributes and cast as tensors"""

    def __init__(self, keys: List[str], id: str = "collate-simple-attrs"):
        super().__init__(
            Collate(keys=keys),
            ApplyToAll(torch.tensor),
            id=id,
        )


class CollateTokens(Sequential):
    """A pipeline to collate token fields (*.input_ids, *.attention_mask)."""

    def __init__(
        self,
        prefix: str,
        *,
        tokenizer: PreTrainedTokenizerFast,
        stride: Optional[int] = None,
        id: Optional[str] = None,
    ):
        super().__init__(
            Collate(keys=[f"{prefix}input_ids", f"{prefix}attention_mask"]),
            ReplaceInKeys(prefix, ""),
            Flatten() if stride else None,
            Lambda(tokenizer.pad),
            Nest(stride=stride) if stride else None,
            AddPrefix(prefix),
            id=id or f"Collate-tokens-{prefix.replace('.', '')}",
        )
