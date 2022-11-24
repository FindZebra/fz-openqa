from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import T
from typing import Union

import torch
from datasets import Dataset
from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast
from warp_pipes import HfDataset


def append_prefix_tokens(
    prefix: Union[str, List[str]],
    text: str,
):
    """
    This functions append a special token to a text such that output = special_token+text.
    The pretrained tokenizer with registered special tokens will encode the output as:
    [CLS][SPEC][ text tokens ][SEP]
    """
    if isinstance(prefix, list):
        prefix = "".join(prefix)
    return f"{prefix}{text}"


def set_index_column(dataset: HfDataset, *, key: str) -> T:
    if isinstance(dataset, DatasetDict):
        return DatasetDict({k: set_index_column(v, key=key) for k, v in dataset.items()})
    elif isinstance(dataset, Dataset):
        return dataset.add_column(key, list(range(dataset.num_rows)))
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")


def set_constant_column(dataset: HfDataset, *, key: str, value: Any):
    if isinstance(dataset, DatasetDict):
        dataset = DatasetDict({k: set_index_column(v, key=key) for k, v in dataset.items()})
    elif isinstance(dataset, Dataset):
        dataset = dataset.add_column(key, len(dataset) * [value])
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    return dataset


def append_document_title(example: Dict[str, Any]) -> Dict[str, Any]:
    example["document"] = f"{example['document.title']}. {example['document']}"
    return example


def truncate_examples_to_max_length(output, *, key: str, tokenizer: PreTrainedTokenizerFast):
    # infer `max_length`
    tokens = [t for t in output[f"{key}.input_ids"]]
    pad_tok = tokenizer.pad_token_id
    max_length = len(tokens[0]) - min(map(lambda x: sum([int(t == pad_tok) for t in x]), tokens))

    # truncate to `max_length`
    def maybe_truncate(x: Any, max_length: int):
        """truncate sequential attributes to `max_length`"""
        if not (isinstance(x, torch.Tensor) and len(x.shape) == 2):
            return x

        return x[:, :max_length]

    tensor_outpus = {k: maybe_truncate(v, max_length) for k, v in output.items()}
    return tensor_outpus
