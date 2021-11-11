from functools import partial
from typing import List
from typing import Optional

from transformers import PreTrainedTokenizerFast

from fz_openqa.datamodules.pipes import AddPrefix
from fz_openqa.datamodules.pipes import Apply
from fz_openqa.datamodules.pipes import ApplyToAll
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import ReplaceInKeys
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.datamodules.pipes import TokenizerPipe
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.datamodules.pipes.utils.nesting import flatten_nested
from fz_openqa.datamodules.pipes.utils.nesting import nested_list
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
        stride: Optional[int],
        return_token_type_ids: bool = False,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        field: str = "text",
    ):

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
            ApplyToAll(flatten_nested, element_wise=False) if stride else None,
            add_spec_tokens,
            TokenizerPipe(
                tokenizer,
                max_length=max_length,
                fields=field,
                return_token_type_ids=return_token_type_ids,
                add_special_tokens=add_special_tokens,
                return_offsets_mapping=return_offsets_mapping,
            ),
            ApplyToAll(partial(nested_list, stride=stride)) if stride else None,
            AddPrefix(prefix),
        )
