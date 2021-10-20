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
from fz_openqa.datamodules.pipes.nesting import flatten_nested
from fz_openqa.datamodules.pipes.nesting import nested_list
from fz_openqa.datamodules.utils.filter_keys import KeyIn
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
    ):

        if add_encoding_tokens:
            add_spec_tokens = Apply(
                {"text": partial(add_spec_token, spec_tokens)},
                element_wise=True,
            )
        else:
            add_spec_tokens = None

        super().__init__(
            FilterKeys(KeyIn([f"{prefix}text"])),
            ReplaceInKeys(prefix, ""),
            text_formatter.copy(text_key="text"),
            ApplyToAll(flatten_nested, element_wise=False) if stride else None,
            add_spec_tokens,
            TokenizerPipe(
                tokenizer,
                max_length=max_length,
                fields="text",
                return_token_type_ids=False,
                add_special_tokens=False,
                return_offsets_mapping=False,
            ),
            ApplyToAll(partial(nested_list, stride=stride))
            if stride
            else None,
            AddPrefix(prefix),
        )
