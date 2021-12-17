from __future__ import annotations

import logging
import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from datasets import DatasetDict
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from fz_openqa.datamodules.builders.base import DatasetBuilder
from fz_openqa.datamodules.pipes import FilterKeys
from fz_openqa.datamodules.pipes import Lambda
from fz_openqa.datamodules.pipes import Pipe
from fz_openqa.datamodules.pipes import Sequential
from fz_openqa.datamodules.pipes import TextFormatter
from fz_openqa.datamodules.pipes import TokenizerPipe
from fz_openqa.datamodules.pipes.control.condition import In
from fz_openqa.datamodules.utils.dataset import get_column_names
from fz_openqa.datamodules.utils.dataset import keep_only_columns
from fz_openqa.datamodules.utils.dataset import take_subset
from fz_openqa.datamodules.utils.typing import HfDataset
from fz_openqa.utils.fingerprint import get_fingerprint
from fz_openqa.utils.pretty import pretty_decode

logger = logging.getLogger(__name__)


def cache_hf_dataset(func):
    """
    Cache the output of the __call__ function by saving the dataset to a file.
    Looks actually slower, but might be useful in a distributed setup.

    Usage:
    ```
    @cache_hf_dataset
    def __call__(self, *args, **kwargs):
        ...
    ```
    """

    def wrapper(self, *args, **kwargs):
        if self._cache_path is None:
            # process dataset
            dataset: HfDataset = func(self, *args, **kwargs)

            if self._cache_dir is not None:
                # get fingerprint
                if isinstance(dataset, DatasetDict):
                    fingerprint = get_fingerprint({k: d._fingerprint for k, d in dataset.items()})
                else:
                    fingerprint = dataset._fingerprint

                # save dataset
                self._cache_path = os.path.join(self._cache_dir, fingerprint)
                self._cache_type = type(dataset)
                if not os.path.exists(self._cache_path):
                    logger.info(f"Writing dataset to {self._cache_path}")
                    dataset.save_to_disk(self._cache_path)
        else:
            logger.info(f"Loading cached dataset from {self._cache_path}")
            dataset: HfDataset = self._cache_type.load_from_disk(self._cache_path)

        return dataset

    return wrapper


class HfDatasetBuilder(DatasetBuilder):
    """This class allows loading a preprocessing a `dataset.Dataset`"""

    # HuggingFace dataset id or local path to script
    dset_script_path_or_id = "ptb_text_only"

    # text field from the raw datasets that should be tokenized
    text_field = "sentence"

    # name of the attributes that will be converted to
    # tensors in the preprocessing function
    pt_attributes = ["input_ids", "attention_mask"]

    # number of data points per subset train/val/test
    subset_size = [100, 10, 10]

    # output columns
    column_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerFast,
        add_encoding_tokens: bool = True,
        add_special_tokens: bool = True,
        cache_dir: str = "cache/",
        max_length: Optional[int] = 512,
        use_subset: bool = False,
        num_proc: int = 1,
        verbose: bool = False,
        text_formatter: Optional[TextFormatter] = None,
        **kwargs,
    ):
        super().__init__(cache_dir=cache_dir, **kwargs)

        self.cache_dir = cache_dir
        self.use_subset = use_subset
        self.num_proc = num_proc
        self.verbose = verbose

        # tokenizer and dataset
        self.text_formatter = text_formatter or TextFormatter()
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.add_encoding_tokens = add_encoding_tokens
        self.add_special_tokens = add_special_tokens

    # @cache_hf_dataset
    def _call(
        self, format: Optional[str] = "torch", columns: Optional[List[str]] = None, **kwargs
    ) -> HfDataset:
        """
        Loads the dataset and preprocesses it
        Parameters
        ----------
        format
            Output format (see `Dataset.set_format`)
        columns
            Columns to include in the output dataset
        kwargs
            Other arguments, unused here.

        Returns
        -------
        HfDataset
            The preprocessed dataset.
        """
        dataset = self.load_and_filter_dataset()
        dataset = self.preprocess_dataset(dataset)
        if format is not None:
            dataset = self.set_format(dataset, format=format)
        return keep_only_columns(dataset, columns=columns)

    def set_format(self, dataset: HfDataset, *, format: str = "torch") -> HfDataset:
        pt_cols = [c for c in self.pt_attributes if c in get_column_names(dataset)]
        dataset.set_format(type=format, columns=pt_cols, output_all_columns=True)
        return dataset

    def load_and_filter_dataset(self) -> HfDataset:
        dataset: HfDataset = self.load_base_dataset()
        dataset = self.filter_dataset(dataset)
        if self.use_subset:
            dataset = take_subset(dataset, self.subset_size)
        return dataset

    def load_base_dataset(self) -> DatasetDict:
        """Load the base HuggingFace dataset."""
        return load_dataset(self.dset_script_path_or_id, cache_dir=self.cache_dir)

    def filter_dataset(self, dataset: HfDataset) -> HfDataset:
        """Apply filter operation to the dataset and return"""
        return dataset

    def preprocess_dataset(self, dataset: HfDataset) -> HfDataset:
        """Apply processing steps to the dataset.
        Tokenization and formatting as PyTorch tensors"""
        pipe = Sequential(
            self.text_formatter.copy(text_key=self.text_field),
            TokenizerPipe(
                self.tokenizer,
                max_length=self.max_length,
                fields=self.text_field,
                return_token_type_ids=False,
                add_special_tokens=False,
                return_offsets_mapping=False,
            ),
        )

        dataset = dataset.map(
            pipe,
            batched=True,
            num_proc=self.num_proc,
            desc="Tokenizing",
            remove_columns=self.text_field,
        )
        return dataset

    def get_collate_pipe(
        self, transform: Optional[Callable | Pipe] = None, columns: Optional[List[str]] = None
    ) -> Pipe:
        pipe = self._get_collate_pipe()
        if columns is not None:
            pipe = Sequential(pipe, FilterKeys(In(columns)))

        if transform is not None:
            pipe = Sequential(pipe, transform)

        return pipe

    def _get_collate_pipe(self) -> Pipe:
        """Returns a pipe that allow collating multiple rows into one Batch"""
        return Sequential(Lambda(self.tokenizer.pad), Lambda(dict))

    def format_row(self, row: Dict[str, Any]) -> str:
        """format a row from the dataset"""
        return pretty_decode(
            row["input_ids"],
            tokenizer=self.tokenizer,
            skip_special_tokens=True,
        )
