from typing import Any
from typing import List
from typing import Optional

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from fz_openqa.datamodules.pipes import AddPrefix
from fz_openqa.datamodules.pipes import ApplyAsFlatten
from fz_openqa.datamodules.pipes import ApplyToAll
from fz_openqa.datamodules.pipes import Collate
from fz_openqa.datamodules.pipes import Gate
from fz_openqa.datamodules.pipes import Lambda
from fz_openqa.datamodules.pipes import Parallel
from fz_openqa.datamodules.pipes import PrintBatch
from fz_openqa.datamodules.pipes import ReplaceInKeys
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes.control.batch_condition import HasKeys
from fz_openqa.datamodules.pipes.control.batch_condition import HasKeyWithPrefix
from fz_openqa.datamodules.pipes.control.condition import Contains
from fz_openqa.datamodules.pipes.control.condition import HasPrefix
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.datamodules.pipes.control.condition import Not
from fz_openqa.datamodules.pipes.control.condition import Reduce


def to_tensor_op(inputs: List[Any]) -> Tensor:
    if all(isinstance(x, Tensor) for x in inputs):
        return torch.cat([t[None] for t in inputs])
    else:
        return torch.tensor(inputs)


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
        if to_tensor is None:
            to_tensor = []

        # pipe us to tensorize values
        if len(to_tensor):
            tensorizer_pipe = ApplyToAll(
                op=to_tensor_op, allow_kwargs=False, input_filter=In(to_tensor)
            )
        else:
            tensorizer_pipe = None

        # pipe used to pad and collate tokens
        if tokenizer is not None:
            tokenizer_pipe = Gate(
                HasKeys(["input_ids"]),
                pipe=ApplyAsFlatten(Lambda(tokenizer.pad), level=level),
                input_filter=In(["input_ids", "attention_mask", "offset_mapping"]),
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
            input_filter=Reduce(HasPrefix(prefix), *[Not(Contains(e)) for e in exclude]),
        )
        super(CollateField, self).__init__(condition=HasKeyWithPrefix(prefix), pipe=pipe, **kwargs)
