import shutil
from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import datasets
import rich
import torch
from datasets import DatasetDict
from datasets import load_dataset
from datasets import Split
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BatchEncoding
from transformers import PreTrainedTokenizerFast

from fz_openqa.utils.datastruct import Batch
from fz_openqa.utils.datastruct import pprint_batch


def collate_and_pad_attributes(
    examples: List[Batch],
    *,
    tokenizer: PreTrainedTokenizerFast,
    key: Optional[str],
    exclude: Optional[str] = None,
) -> Batch:
    """
    Collate the input encodings for a given key (e.g. "document", "question", ...)
    using `PreTrainedTokenizerFast.pad`. Check the original documentation to see what types are
    compatible. Return a Batch {'key.x' : ...., 'key.y': ....}
    """

    # remove the key, so the tokenizer receives the attributes without the key prefix
    tokenizer_inputs = [
        {
            k.replace(f"{key}.", ""): v
            for k, v in ex.items()
            if (key is not None and f"{key}." in k)
            and (exclude is None or exclude not in k)
        }
        for ex in examples
    ]

    # collate using the tokenizer
    output = tokenizer.pad(tokenizer_inputs)

    # re-append the key prefix
    return {f"{key}.{k}": v for k, v in output.items()}
