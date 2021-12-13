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
        text_formatter: Optional[TextFormatter] = None,
        tokenizer: PreTrainedTokenizerFast,
        add_encoding_tokens: bool = True,
        max_length: Optional[int] = 512,
        spec_token: Optional[str] = None,
        shape: Optional[List[int]] = None,
        return_token_type_ids: bool = False,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        key: str = "text",
    ):
        if shape is None:
            shape = [-1]

        if add_encoding_tokens and spec_token is not None:
            add_spec_tokens_pipe = Apply(
                {key: partial(add_spec_token, spec_token)},
                element_wise=True,
            )
        else:
            add_spec_tokens_pipe = None

        if text_formatter is not None:
            text_formatter = text_formatter.copy(text_key=key)

        super().__init__(
            FilterKeys(In([f"{prefix}{key}"])),
            ReplaceInKeys(prefix, ""),
            text_formatter,
            ApplyAsFlatten(
                Sequential(
                    add_spec_tokens_pipe,
                    TokenizerPipe(
                        tokenizer,
                        max_length=max_length,
                        fields=key,
                        return_token_type_ids=return_token_type_ids,
                        add_special_tokens=add_special_tokens,
                        return_offsets_mapping=return_offsets_mapping,
                    ),
                ),
                level=len(shape) - 1,
            ),
            AddPrefix(prefix),
        )
