from __future__ import annotations  # noqa: F407

import re
from functools import partial
from typing import List
from typing import Optional

from transformers import PreTrainedTokenizerFast
from warp_pipes import AddPrefix
from warp_pipes import Apply
from warp_pipes import ApplyAsFlatten
from warp_pipes import Batch
from warp_pipes import FilterKeys
from warp_pipes import Pipe
from warp_pipes import ReplaceInKeys
from warp_pipes import Sequential
from warp_pipes import TokenizerPipe
from warp_pipes.core.condition import In

from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.datamodules.utils.transformations import append_prefix


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
        add_qad_tokens: bool = True,
        max_length: Optional[int] = 512,
        qad_tokens: Optional[str | List[str]] = None,
        shape: Optional[List[int]] = None,
        return_token_type_ids: bool = False,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        key: str = "text",
        **kwargs,
    ):
        if shape is None:
            shape = [-1]

        if isinstance(qad_tokens, str):
            qad_tokens = [qad_tokens]

        if add_qad_tokens and qad_tokens is not None:
            qad_tokens = "".join(qad_tokens)
            add_qad_tokens_pipe = Apply(
                {key: partial(append_prefix, qad_tokens)},
                element_wise=True,
            )
        else:
            add_qad_tokens_pipe = None

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
                    add_qad_tokens_pipe,
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


class AppendSuffix(Pipe):
    def __init__(self, text_fields: str | List[str], suffix: str = ". ", **kwargs):
        super().__init__(**kwargs)
        if isinstance(text_fields, str):
            text_fields = [text_fields]
        self.text_fields = text_fields
        self.suffix = suffix

    def add_dot(self, x: str) -> str:
        if x is not None and len(x):
            return f"{x}{self.suffix}"
        elif x is None:
            return ""
        else:
            return x

    def _call_batch(self, batch: Batch, text_fields: str = "title", **kwargs) -> Batch:
        out = {}
        for text_field in self.text_fields:
            out[text_field] = list(map(self.add_dot, batch[text_field]))

        return out
