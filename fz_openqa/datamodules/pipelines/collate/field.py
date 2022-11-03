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


class CollateField(Gate):
    """
    Collate examples for a given field.
    Field corresponds to the prefix of the keys (field.attribute)
    This Pipe is a Gate and is only activated if keys for the field are present.

    This class handles nested examples, which nesting level must be indicated using `level`.
    """

    def __init__(
        self,
        field: str,
        *,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        exclude: Optional[List[str]] = None,
        include_only: Optional[List[str]] = None,
        to_tensor: Optional[List[str]] = None,
        level: int = 0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        field
            Field to collate
        tokenizer
            Tokenizer to use for the input_ids and attention_mask attributes
        exclude
            List of attribute to exclude
        to_tensor
            List of attribute to convert to tensor
        level
            Nesting level to consider
        kwargs
            Additional arguments to pass to the base class
        """

        # set defaults
        prefix = f"{field}."
        if exclude is None:
            exclude = []
        if include_only is None:
            include_only = []
        if to_tensor is None:
            to_tensor = []

        # define the input filter
        if len(include_only):
            include_keys = [f"{prefix}{i}" for i in include_only]
            include_only_cond = In(include_keys)
        else:
            include_only_cond = True

        input_filter = Reduce(
            HasPrefix(prefix),
            include_only_cond,
            *[Not(Contains(f"{prefix}{e}")) for e in exclude],
            reduce_op=all,
        )

        # pipe us to tensorize values
        if len(to_tensor):
            tensorizer_pipe = ApplyToAll(
                op=to_tensor_op, allow_kwargs=False, input_filter=In(to_tensor)
            )
            tensorizer_pipe = ApplyAsFlatten(tensorizer_pipe, level=level)
        else:
            tensorizer_pipe = None

        # pipe used to pad and collate tokens
        if tokenizer is not None:
            tokenizer_pipe = Gate(
                HasKeys(["input_ids"]),
                pipe=ApplyAsFlatten(Padding(tokenizer=tokenizer), level=level),
                input_filter=In(
                    ["input_ids", "attention_mask", "offset_mapping", "token_type_ids"]
                ),
                id="pad-and-collate-tokens",
            )
        else:
            tokenizer_pipe = None

        # define the body of the pipe, all pipe bellow operates without the prefix.
        if tokenizer_pipe is not None or tensorizer_pipe is not None:
            body = Sequential(
                ReplaceInKeys(prefix, ""),
                Parallel(tensorizer_pipe, tokenizer_pipe, update=True),
                AddPrefix(prefix),
            )
        else:
            body = None

        # full pipe
        pipe = Sequential(
            Collate(),
            body,
            input_filter=input_filter,
        )

        super(CollateField, self).__init__(condition=HasKeyWithPrefix(prefix), pipe=pipe, **kwargs)
