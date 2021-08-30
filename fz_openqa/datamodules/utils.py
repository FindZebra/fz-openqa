from typing import Any
from typing import List

from fz_openqa.tokenizers.static import ANS_TOKEN
from fz_openqa.tokenizers.static import DOC_TOKEN
from fz_openqa.tokenizers.static import QUERY_TOKEN


def nested_list(values: List[Any], *, stride: int) -> List[List[Any]]:
    output = []
    for i in range(0, len(values), stride):
        output += [[values[j] for j in range(i, i + stride)]]
    return output


def add_spec_token(
    special_token: str,
    text: str,
):
    """
    This functions append a special token to a text such that output = special_token+text.
    The pretrained tokenizer with registered special tokens will encode the output as:
    [CLS][SPEC][ text tokens ][SEP]
    """
    assert special_token in [QUERY_TOKEN, ANS_TOKEN, DOC_TOKEN]
    return f"{special_token}{text}"
