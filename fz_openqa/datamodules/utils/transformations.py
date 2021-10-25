from typing import Any
from typing import Dict

import torch
from transformers import PreTrainedTokenizerFast

from fz_openqa.tokenizers.static import ANS_TOKEN
from fz_openqa.tokenizers.static import DOC_TOKEN
from fz_openqa.tokenizers.static import QUERY_TOKEN


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


def set_row_idx(
    example: Dict[str, Any], idx: int, key: str = "idx"
) -> Dict[str, Any]:
    example[key] = idx
    return example


def append_document_title(example: Dict[str, Any]) -> Dict[str, Any]:
    example["document"] = f"{example['document.title']}. {example['document']}"
    return example


def truncate_examples_to_max_length(
    output, *, key: str, tokenizer: PreTrainedTokenizerFast
):
    # infer `max_length`
    tokens = [t for t in output[f"{key}.input_ids"]]
    pad_tok = tokenizer.pad_token_id
    max_length = len(tokens[0]) - min(
        map(lambda x: sum([int(t == pad_tok) for t in x]), tokens)
    )

    # truncate to `max_length`
    def maybe_truncate(x: Any, max_length: int):
        """truncate sequential attributes to `max_length`"""
        if not (isinstance(x, torch.Tensor) and len(x.shape) == 2):
            return x

        return x[:, :max_length]

    tensor_outpus = {
        k: maybe_truncate(v, max_length) for k, v in output.items()
    }
    return tensor_outpus
