from functools import partial
from typing import List
from typing import Optional

from transformers import PreTrainedTokenizerFast

from fz_openqa.datamodules.pipes import AddPrefix
from fz_openqa.datamodules.pipes import Apply
from fz_openqa.datamodules.pipes import ApplyAsFlatten
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import ReplaceInKeys
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.datamodules.pipes import TokenizerPipe
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.datamodules.utils.transformations import add_spec_token


class FormatAndTokenize(Sequential):
    """
    create a pipeline to process the raw text:
        1. format text
        2. add special tokens
        3. tokenize
    """

    def __init__(
        self,
        prefix: str,
        *,
        text_formatter: TextFormatter,
        tokenizer: PreTrainedTokenizerFast,
        add_encoding_tokens: bool,
        max_length: Optional[int],
        spec_tokens: List,
        shape: Optional[List[int]],
        return_token_type_ids: bool = False,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        field: str = "text",
    ):
        if shape is None:
            shape = [-1]

        if add_encoding_tokens:
            add_spec_tokens = Apply(
                {"text": partial(add_spec_token, spec_tokens)},
                element_wise=True,
            )
        else:
            add_spec_tokens = None

        super().__init__(
            FilterKeys(In([f"{prefix}text"])),
            ReplaceInKeys(prefix, ""),
            text_formatter.copy(text_key="text"),
            ApplyAsFlatten(
                Sequential(
                    add_spec_tokens,
                    TokenizerPipe(
                        tokenizer,
                        max_length=max_length,
                        fields=field,
                        return_token_type_ids=return_token_type_ids,
                        add_special_tokens=add_special_tokens,
                        return_offsets_mapping=return_offsets_mapping,
                    ),
                ),
                level=len(shape) - 1,
            ),
            AddPrefix(prefix),
        )
