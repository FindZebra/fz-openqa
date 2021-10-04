from typing import Any
from typing import Dict
from typing import List
from typing import Union

import torch
from datasets import Dataset
from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast

from fz_openqa.tokenizers.static import ANS_TOKEN
from fz_openqa.tokenizers.static import DOC_TOKEN
from fz_openqa.tokenizers.static import QUERY_TOKEN

HgDataset = Union[Dataset, DatasetDict]


def get_column_names(dataset: HgDataset) -> List[str]:
    if isinstance(dataset, DatasetDict):
        return list(
            set.union(*(set(d.column_names) for d in dataset.values()))
        )
    else:
        return dataset.column_names


def flatten_nested(values: List[List[Any]]) -> List[Any]:
    return [sub_x for x in values for sub_x in x]


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


def take_subset(dataset: HgDataset, subset_size: List[int]) -> HgDataset:
    """Take a subset of the dataset and return."""
    if isinstance(dataset, DatasetDict):
        return DatasetDict(
            {
                k: dset.select(range(n))
                for n, (k, dset) in zip(subset_size, dataset.items())
            }
        )
    elif isinstance(dataset, Dataset):
        size = next(iter(subset_size))
        return dataset.select(range(size))
    else:
        raise NotImplementedError


def set_example_idx(
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
