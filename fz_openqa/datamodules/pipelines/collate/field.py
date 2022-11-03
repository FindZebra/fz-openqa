from copy import copy
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import rich
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerFast
from warp_pipes import AddPrefix
from warp_pipes import ApplyAsFlatten
from warp_pipes import ApplyToAll
from warp_pipes import Batch
from warp_pipes import Collate
from warp_pipes import Gate
from warp_pipes import HasKeys
from warp_pipes import HasKeyWithPrefix
from warp_pipes import Lambda
from warp_pipes import Parallel
from warp_pipes import Pipe
from warp_pipes import pprint_batch
from warp_pipes import ReplaceInKeys
from warp_pipes import Sequential
from warp_pipes.core.condition import Contains
from warp_pipes.core.condition import HasPrefix
from warp_pipes.core.condition import In
from warp_pipes.core.condition import Not
from warp_pipes.core.condition import Reduce


def to_tensor_op(inputs: List[Any]) -> Tensor:
    if all(isinstance(x, Tensor) for x in inputs):
        return torch.cat([t[None] for t in inputs])
    else:
        return torch.tensor(inputs)


class Padding(Pipe):
    def __init__(
        self, *, tokenizer: PreTrainedTokenizerFast, special_padding_tokens: Dict = None, **kwargs
    ):
        super(Padding, self).__init__(**kwargs)
        if special_padding_tokens is None:
            special_padding_tokens = {}
        self.special_padding_tokens = special_padding_tokens
        self.tokenizer = tokenizer

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:

        special_values = {
            k: batch[k] for k in self.special_padding_tokens.keys() if k in batch.keys()
        }
        pad_input = {k: v for k, v in batch.items() if k not in special_values}
        # pad `normal` values using `tokenizer.pad`
        output = self.tokenizer.pad(pad_input, return_tensors="pt")

        # pad the special cases
        for k, v in special_values.items():
            lenght = output["input_ids"].shape[-1]
            output[k] = self._pad(v, lenght, self.special_padding_tokens[k])

        return output

    def _pad(self, x, length, fill_value):
        y = []
        for z in x:
            if len(z) < length:
                z = z + (length - len(z)) * [fill_value]
            y += [torch.tensor(z)]

        return torch.stack(y)
