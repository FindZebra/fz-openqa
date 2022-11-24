from typing import List
from typing import Optional

import torch
from transformers import PreTrainedTokenizerFast
from warp_pipes import AddPrefix
from warp_pipes import AllValuesOfType
from warp_pipes import ApplyAsFlatten
from warp_pipes import ApplyToAll
from warp_pipes import Collate
from warp_pipes import Gate
from warp_pipes import Lambda
from warp_pipes import ReplaceInKeys
from warp_pipes import Sequential


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
        shape: Optional[List[int]] = None,
        id: Optional[str] = None,
    ):
        if shape is None:
            shape = [-1]
        super().__init__(
            Collate(keys=[f"{prefix}input_ids", f"{prefix}attention_mask"]),
            ReplaceInKeys(prefix, ""),
            ApplyAsFlatten(
                Gate(
                    AllValuesOfType(list),
                    pipe=Lambda(tokenizer.pad, id=f"collate:pad:{prefix}"),
                    update=True,
                ),
                level=len(shape) - 1,
            ),
            AddPrefix(prefix),
            id=id or f"Collate-tokens-{prefix.replace('.', '')}",
        )
