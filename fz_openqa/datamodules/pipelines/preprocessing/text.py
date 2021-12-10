from __future__ import annotations

import re
from functools import partial
from typing import List
from typing import Optional

import rich
from transformers import PreTrainedTokenizerFast

from fz_openqa.datamodules.pipes import AddPrefix
from fz_openqa.datamodules.pipes import Apply
from fz_openqa.datamodules.pipes import ApplyAsFlatten
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import ReplaceInKeys
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.datamodules.pipes import TokenizerPipe
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.datamodules.utils.transformations import add_spec_token
from fz_openqa.utils.datastruct import Batch


class CleanupSpecialTokens(Pipe):
    def __init__(self, text_fields: str | List[str], tokenizer: PreTrainedTokenizerFast, **kwargs):
        super().__init__(**kwargs)
        if isinstance(text_fields, str):
            text_fields = [text_fields]
        self.text_fields = text_fields

        # assumes all the special tokens to be in brackets
        all_tokens = [u.replace("[", "").replace("]", "") for u in tokenizer.all_special_tokens]
        self.pattern = r"|".join(map(r"(?:\[{}\])".format, all_tokens))

    def clean(self, x: str) -> str:
        return re.sub(self.pattern, "", x)

    def _call_batch(self, batch: Batch, **kwargs) -> Batch:
        out = {}
        for text_field in self.text_fields:
            out[text_field] = list(map(self.clean, batch[text_field]))

        return out


class FormatAndTokenize(Sequential):
    """
    create a pipeline to process the raw text:
        1. format text
        2. add special tokens
        3. tokenize
    """

    def __init__(
        self,
        prefix: Optional[str],
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
        **kwargs,
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

        # define the text formatter, used to cleanup the raw text
        if text_formatter is not None:
            text_formatter = text_formatter.copy(text_key=key)

        # define pipes to filter by prefix
        if prefix is not None:
            format_out = AddPrefix(prefix)
            format_in = Sequential(FilterKeys(In([f"{prefix}{key}"])), ReplaceInKeys(prefix, ""))

        else:
            format_out = None
            format_in = None

        super().__init__(
            format_in,
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
            format_out,
            **kwargs,
        )
