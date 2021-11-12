from typing import List
from typing import Optional

import torch
from transformers import PreTrainedTokenizerFast

from fz_openqa.datamodules.pipes import AddPrefix
from fz_openqa.datamodules.pipes import ApplyAsFlatten
from fz_openqa.datamodules.pipes import ApplyToAll
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes import Gate
from fz_openqa.datamodules.pipes import Lambda
from fz_openqa.datamodules.pipes import ReplaceInKeys
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes.control.batch_condition import AllValuesOfType


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
