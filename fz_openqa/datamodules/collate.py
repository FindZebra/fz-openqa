from collections import defaultdict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from transformers import PreTrainedTokenizerFast

from fz_openqa.utils.datastruct import Batch


def collate_and_pad_attributes(
    examples: List[Batch],
    *,
    tokenizer: PreTrainedTokenizerFast,
    key: Optional[str],
    exclude: Optional[Union[str, List[str]]] = None,
) -> Batch:
    """
    Collate the input encodings for a given key (e.g. "document", "question", ...)
    using `PreTrainedTokenizerFast.pad`. Check the original documentation to see what types are
    compatible. Return a Batch {'key.x' : ...., 'key.y': ....}
    """
    if isinstance(exclude, str):
        exclude = [exclude]
    if exclude is None:
        exclude = []

    # remove the key, so the tokenizer receives the attributes without the key prefix
    tokenizer_inputs = [
        {
            k.replace(f"{key}.", ""): v
            for k, v in ex.items()
            if (key is not None and f"{key}." in k)
            and (len(exclude) == 0 or all(ex not in k for ex in exclude))
        }
        for ex in examples
    ]

    # collate using the tokenizer
    output = tokenizer.pad(tokenizer_inputs)

    # re-append the key prefix
    return {f"{key}.{k}": v for k, v in output.items()}


def extract_and_collate_attributes_as_list(
    examples: List[Batch],
    *,
    attribute: str,
) -> Tuple[List[Batch], Batch]:
    """
    Extract the attribute fields (e.g. `document.text`) from a list of Examples
    and return all fields as a Batch `{'document.{attribute}': ["...", "..."]}`.
    The target attributes are removed from the original examples
    """
    text_keys = [key for key in examples[0].keys() if f".{attribute}" in key]
    text_outputs = defaultdict(list)
    for ex in examples:
        for key in text_keys:
            text_outputs[key] += [ex.pop(key)]

    return examples, text_outputs
